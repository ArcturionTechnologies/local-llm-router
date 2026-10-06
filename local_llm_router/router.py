"""The router: pick a tier, gate it, call it, fall through or escalate.

Life of one call to :meth:`Router.complete`:

1. **Pick** a starting tier -- explicitly (``tier="medium"``) or by task
   signature (:func:`pick_tier`).
2. Build the **chain**: that tier followed by its ``fallbacks`` (transitively,
   de-duplicated). Fallbacks are how a call steps *down* under thermal pressure
   and *up* to a bigger/cloud tier when a local one is busy, down or unsuitable.
3. For each tier in the chain:

   * local tier  -> thermal gate (and RAM-pressure demotion for heavy tiers),
     then the model lock if the tier is ``exclusive``;
   * cloud tier  -> API key present? rate-limited recently? budget envelope?
   * send the request; on any :class:`TierError` / gate refusal move to the next tier;
   * if the caller passed ``accept=`` and the answer is rejected, treat that as
     "this tier was not good enough" and **escalate** to the next tier.

4. Record telemetry + budget for every attempt. If the chain is exhausted raise
   :class:`AllTiersFailed` (or re-raise the first error with ``fallback="raise"``,
   or return ``""`` with ``fallback="skip"``).
"""

from __future__ import annotations

import json
import os
import re
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional

from . import client as _client
from .budget import Budget
from .config import Config, load_config
from .errors import (AllTiersFailed, BudgetExceeded, RateLimited, RouterError,
                     ThermalBlocked, ThermalDowngrade, TierBusy, TierError)
from .model_lock import ModelLock
from .telemetry import Telemetry
from .thermal import ThermalGate

RATE_LIMIT_COOLDOWN_S = 600


@dataclass
class Attempt:
    tier: str
    ok: bool
    error: str = ""
    detail: str = ""
    latency_ms: int = 0


@dataclass
class Completion:
    text: str
    tier: Optional[str]
    model: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    latency_ms: int = 0
    attempts: list = field(default_factory=list)
    requested_tier: str = ""

    @property
    def escalated(self) -> bool:
        """True when a tier other than the requested one answered."""
        return bool(self.tier) and self.tier != self.requested_tier

    def __str__(self) -> str:
        return self.text


# --------------------------------------------------------------------------- #
# Tier selection by task signature
# --------------------------------------------------------------------------- #

def pick_tier(config: Config, task_class: str = "other", complexity: str = "simple",
              stakes: str = "low", bulk_count: int = 1, context_tokens: int = 0) -> tuple:
    """Return ``(tier_name, justification)`` for a task signature.

    ``task_class``: classify | extract | summarize | route | reason | code-gen |
    edit | refactor | analyze | review | vision | other.
    ``complexity``: trivial | simple | moderate | complex.
    ``stakes``: low | medium | high | irreversible.

    Hard rules win over everything, in this order: irreversible stakes go to the
    strongest tier; very long contexts go to the long-context tier; bulk loops
    never touch the paid tiers. After that, local-first.
    """
    r = config.role
    tc = (task_class or "other").lower()
    cx = (complexity or "simple").lower()
    sk = (stakes or "low").lower()

    if sk == "irreversible":
        return r("strongest"), "irreversible stakes -- strongest tier required"
    if context_tokens > config.routing.long_context_tokens:
        return r("long_context"), f"context={context_tokens} tokens -- long-context tier"
    if bulk_count > config.routing.bulk_threshold:
        if cx in ("trivial", "simple"):
            return r("bulk_simple"), f"bulk={bulk_count} simple -- free bulk tier"
        return r("bulk_complex"), f"bulk={bulk_count} {cx} -- free large-model tier"

    if tc in ("vision", "ocr", "screenshot"):
        return r("vision"), "vision task -- local vision model"
    if tc in ("classify-fast", "classify", "route") and cx in ("trivial", "simple"):
        return r("fast"), "snap-fast classification -- small local model"
    if tc in ("summarize-short", "summarize") and cx == "trivial":
        return r("fast"), "short summary -- small local model"
    if tc == "extract" and cx in ("trivial", "simple"):
        return r("small"), "trivial extraction -- local model"
    if tc == "summarize" and cx == "simple":
        return r("small"), "simple summary -- local model"
    if tc in ("code-gen", "edit", "refactor"):
        if sk == "high":
            return r("strong"), "high-stakes code -- strong cloud tier"
        return r("coder"), f"{cx} code -- local coder model"
    if tc in ("reason", "analyze"):
        if cx in ("simple", "trivial"):
            return r("small"), "simple reasoning -- local model"
        if sk == "high":
            return r("strong"), "complex high-stakes reasoning -- strong cloud tier"
        return r("medium"), f"{cx} reasoning -- local mid-size model"
    if tc == "review" and sk == "high":
        return r("strongest"), "high-stakes review -- strongest tier"
    if tc == "review":
        return r("medium"), f"{cx} review -- local mid-size model"
    if cx == "complex":
        if sk == "high":
            return r("strongest"), "complex + high-stakes -- strongest tier"
        return r("medium"), "complex reasoning -- local mid-size model"
    if sk == "low":
        return r("fast"), "default low-stakes -- small local model"
    return r("strong"), "default medium-stakes -- strong cloud tier"


# --------------------------------------------------------------------------- #
# Router
# --------------------------------------------------------------------------- #

class Router:
    def __init__(self, config: Optional[Config] = None, *,
                 transport: Optional[_client.Transport] = None,
                 thermal: Optional[ThermalGate] = None,
                 lock: Optional[ModelLock] = None,
                 budget: Optional[Budget] = None,
                 telemetry: Optional[Telemetry] = None,
                 env: Optional[Mapping[str, str]] = None):
        self.config = config or load_config()
        self.env = os.environ if env is None else env
        self.transport = transport or _client.urllib_transport
        self.thermal = thermal or ThermalGate(self.config.thermal, env=self.env)
        self.lock = lock or ModelLock(self.config.lock_dir)
        self.budget = budget or Budget(self.config)
        self.telemetry = telemetry or Telemetry(
            self.config.telemetry_path, self.config.telemetry.enabled, env=self.env)
        self._rl_path = self.config.state_path / "rate-limited.json"

    # -- selection ----------------------------------------------------------
    def pick(self, task_class: str = "other", complexity: str = "simple", stakes: str = "low",
             bulk_count: int = 1, context_tokens: int = 0) -> tuple:
        return pick_tier(self.config, task_class, complexity, stakes, bulk_count, context_tokens)

    def chain(self, name: str) -> list:
        """``name`` followed by its fallbacks, depth-first, without repeats.

        Depth-first keeps each tier's own fallback ahead of its siblings: with
        ``heavy -> [medium, groq]`` and ``medium -> [small, groq]`` the order is
        ``heavy, medium, small, groq`` -- the lighter local steps are tried
        before going to the cloud.
        """
        order, seen, stack = [], set(), [name]
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            tier = self.config.tier(n)
            if tier.enabled:
                order.append(tier)
            stack.extend(reversed(tier.fallbacks))
        return order

    # -- cloud rate-limit memory -------------------------------------------
    def _rate_limits(self) -> dict:
        try:
            return json.loads(self._rl_path.read_text())
        except (OSError, ValueError):
            return {}

    def _is_rate_limited(self, tier: str) -> bool:
        return time.time() < self._rate_limits().get(tier, 0)

    def _mark_rate_limited(self, tier: str, seconds: int = RATE_LIMIT_COOLDOWN_S) -> None:
        data = self._rate_limits()
        data[tier] = time.time() + seconds
        try:
            self._rl_path.parent.mkdir(parents=True, exist_ok=True)
            self._rl_path.write_text(json.dumps(data))
        except OSError:
            pass

    # -- one attempt --------------------------------------------------------
    def _attempt(self, tier, messages, max_tokens, temperature, timeout):
        if tier.kind == "local":
            self.thermal.enforce(tier.profile)
            if tier.demote_on_pressure and self.thermal.should_demote():
                raise ThermalDowngrade("RAM/concurrency pressure -- step down from a heavy tier")
        else:
            if self._is_rate_limited(tier.name):
                raise RateLimited(f"{tier.name}: in local rate-limit cool-down")
            self.budget.enforce(tier.name)
        guard = (self.lock.hold(tier.lock_name)
                 if tier.exclusive and tier.kind == "local" else nullcontext())
        try:
            with guard:
                return _client.complete(tier, messages, max_tokens=max_tokens,
                                        temperature=temperature, timeout=timeout,
                                        transport=self.transport, env=self.env)
        except RateLimited:
            self._mark_rate_limited(tier.name)
            raise

    # -- main entry ---------------------------------------------------------
    def complete(self, prompt, *, system: Optional[str] = None, tier: Optional[str] = None,
                 task_class: str = "other", complexity: str = "simple", stakes: str = "low",
                 bulk_count: int = 1, context_tokens: int = 0,
                 max_tokens: int = 400, temperature: float = 0.2, timeout: Optional[float] = None,
                 fallback: str = "cascade", accept: Optional[Callable[[str], bool]] = None,
                 images: Optional[list] = None, task: str = "chat") -> Completion:
        """Run one completion through the tier chain. See the module docstring.

        ``prompt`` is a string or a ready-made list of chat messages.
        ``fallback``: ``"cascade"`` (default) walk the chain; ``"raise"`` try only
        the requested tier and re-raise its error; ``"skip"`` return an empty
        :class:`Completion` instead of raising when everything fails.
        """
        if fallback not in ("cascade", "raise", "skip"):
            raise ValueError(f"unknown fallback: {fallback!r}")
        if tier is None:
            tier, _why = self.pick(task_class, complexity, stakes, bulk_count, context_tokens)
        messages = self._messages(prompt, system, images)
        chain = self.chain(tier)
        if not chain:
            raise AllTiersFailed(f"tier {tier!r} and its fallbacks are all disabled", [])

        attempts: list = []
        first_error: Optional[Exception] = None
        for idx, t in enumerate(chain):
            if fallback == "raise" and idx > 0:
                break
            started = time.time()
            try:
                text, usage = self._attempt(t, messages, max_tokens, temperature,
                                            timeout if idx == 0 else None)
            except (TierError, ThermalBlocked, ThermalDowngrade, BudgetExceeded) as e:
                ms = int((time.time() - started) * 1000)
                attempts.append(Attempt(t.name, False, type(e).__name__, str(e), ms))
                self._log(t, task, prompt, "", ms, False, error=type(e).__name__,
                          fallback=idx > 0, usage=None)
                first_error = first_error or e
                continue
            ms = int((time.time() - started) * 1000)
            tin = usage.tokens_in if usage.real else _client.estimate_tokens(messages)
            tout = usage.tokens_out if usage.real else (_client.estimate_tokens(text) if text else 0)
            if accept is not None and not accept(text):
                attempts.append(Attempt(t.name, False, "rejected", "answer rejected by accept()", ms))
                self._log(t, task, prompt, text, ms, True, error="rejected", fallback=idx > 0,
                          usage=(tin, tout))
                self.budget.record(t.name, tin, tout)
                first_error = first_error or TierError(f"{t.name}: answer rejected")
                continue
            attempts.append(Attempt(t.name, True, latency_ms=ms))
            self._log(t, task, prompt, text, ms, True, fallback=idx > 0, usage=(tin, tout))
            self.budget.record(t.name, tin, tout)
            return Completion(text=text, tier=t.name, model=t.model, tokens_in=tin,
                              tokens_out=tout, latency_ms=ms, attempts=attempts,
                              requested_tier=tier)

        if fallback == "raise" and first_error is not None:
            raise first_error
        if fallback == "skip":
            return Completion(text="", tier=None, attempts=attempts, requested_tier=tier)
        trail = "; ".join(f"{a.tier}: {a.error or 'ok'}" for a in attempts)
        raise AllTiersFailed(f"all tiers failed ({trail})", attempts)

    def chat(self, prompt, **kw) -> str:
        """:meth:`complete` returning just the text."""
        return self.complete(prompt, **kw).text

    # -- helpers ------------------------------------------------------------
    @staticmethod
    def _messages(prompt, system, images) -> list:
        if isinstance(prompt, list):
            msgs = list(prompt)
            if system:
                msgs.insert(0, {"role": "system", "content": system})
            return msgs
        msgs = [{"role": "system", "content": system}] if system else []
        if images:
            content = [{"type": "text", "text": prompt}] + [_client.image_part(p) for p in images]
        else:
            content = prompt
        msgs.append({"role": "user", "content": content})
        return msgs

    def _log(self, tier, task, prompt, response, latency_ms, ok, *, error="",
             fallback=False, usage=None) -> None:
        rec = {"tier": tier.name, "provider": tier.kind, "cost": tier.cost_class,
               "model": tier.model, "task": task, "latency_ms": latency_ms, "ok": ok}
        if usage:
            rec["tokens_in"], rec["tokens_out"] = usage
        else:
            rec["tokens_in"] = _client.estimate_tokens(prompt) if prompt else 0
            rec["tokens_out"] = 0
        if error:
            rec["error"] = error
        if fallback:
            rec["fallback"] = True
        self.telemetry.log(**rec)

    def health(self, timeout: float = 1.5) -> dict:
        """``{tier: bool}`` for every enabled tier (local: GET /models; cloud: key present)."""
        return {t.name: _client.health(t, timeout=timeout, transport=self.transport, env=self.env)
                for t in self.config.tiers.values() if t.enabled}

    # -- convenience tasks --------------------------------------------------
    def _helper_tier(self, tier: Optional[str]) -> str:
        return tier or self.config.role("helper")

    def classify(self, text: str, labels: list, *, tier: Optional[str] = None,
                 timeout: Optional[float] = None) -> str:
        """Pick exactly one label. Always returns one of ``labels`` (the last on junk)."""
        if not labels:
            raise ValueError("labels required")
        sys_msg = ("You are a strict classifier. Read the input and respond with EXACTLY one "
                   f"of these labels and nothing else: {', '.join(labels)}")
        out = self.chat(text, system=sys_msg, tier=self._helper_tier(tier), max_tokens=20,
                        temperature=0.0, timeout=timeout, task="classify")
        out = out.strip().strip(".").strip('"').strip("'").lower()
        for lab in labels:                      # exact match first...
            if lab.lower() == out:
                return lab
        for lab in labels:                      # ...then whole-word, so "earn" can't shadow "earnings"
            if re.search(r"\b" + re.escape(lab.lower()) + r"\b", out):
                return lab
        return labels[-1]

    def extract_json(self, text: str, schema_hint: dict, *, tier: Optional[str] = None,
                     timeout: Optional[float] = None) -> dict:
        """Extract structured JSON. ``schema_hint`` is ``{field: type_string}``."""
        fields = ", ".join(f"{k}:{v}" for k, v in schema_hint.items())
        sys_msg = ("Extract structured data from the input. Respond ONLY with valid JSON, "
                   f"no prose, no markdown fences. Required fields: {fields}. "
                   "Use null when a field is not present.")
        raw = self.chat(text, system=sys_msg, tier=self._helper_tier(tier), max_tokens=500,
                        temperature=0.0, timeout=timeout, task="extract").strip()
        fenced = re.match(r"^```[a-zA-Z]*\s*\n?(.*?)\n?```\s*$", raw, re.DOTALL)
        if fenced:
            raw = fenced.group(1).strip()
        try:
            return json.loads(raw)
        except ValueError:
            start, end = raw.find("{"), raw.rfind("}")
            if start >= 0 and end > start:
                try:
                    return json.loads(raw[start:end + 1])
                except ValueError:
                    pass
        raise TierError(f"no JSON in response: {raw[:200]}")

    def summarize(self, text: str, max_words: int = 40, *, tier: Optional[str] = None,
                  timeout: Optional[float] = None) -> str:
        sys_msg = (f"Summarize the input in at most {max_words} words. Plain prose, no bullets, "
                   "no preamble. Capture the key fact and any number/date/name mentioned.")
        return self.chat(text, system=sys_msg, tier=self._helper_tier(tier),
                         max_tokens=max(80, max_words * 2), temperature=0.1,
                         timeout=timeout, task="summarize")

    def route(self, prompt: str, options: list, **kw) -> str:
        """Pick which agent/tool/route should handle ``prompt``."""
        return self.classify(prompt, options, **kw)

    def should_escalate(self, prompt: str, *, tier: Optional[str] = None,
                        timeout: Optional[float] = None) -> bool:
        """Cheap pre-check: can a small local model handle this, or is it worth a bigger tier?"""
        sys_msg = ("Decide if a request needs a frontier LLM or can be handled by a small local "
                   "model. Reply with one word: ESCALATE if it needs reasoning, code generation, "
                   "or multi-step planning. LOCAL if it's classification, summarization, "
                   "extraction, or a yes/no judgment.")
        out = self.chat(prompt, system=sys_msg, tier=self._helper_tier(tier), max_tokens=5,
                        temperature=0.0, timeout=timeout, task="should_escalate")
        return "ESCALATE" in out.upper()

    def vision(self, prompt: str, images: list, **kw) -> str:
        """Ask the vision tier about local image files (inlined as data URLs)."""
        kw.setdefault("tier", self.config.role("vision"))
        kw.setdefault("fallback", "raise")
        kw.setdefault("max_tokens", 512)
        return self.chat(prompt, images=images, task="vision", **kw)

    # -- tier escalation by complexity -------------------------------------
    def classify_complexity(self, prompt: str) -> str:
        """Ask a small local model: ``trivial``, ``routine`` or ``hard``?

        Falls back to ``routine`` if no local tier can answer -- classification
        must never block the real work.
        """
        try:
            out = self.complete(
                f"Classify this LLM task complexity:\n\n{prompt[:1500]}",
                system=("You are a strict classifier. Respond with EXACTLY one of these labels "
                        "and nothing else: trivial, routine, hard"),
                tier=self.config.role("fast"), fallback="raise", max_tokens=8,
                temperature=0.0, task="complexity").text.strip().lower()
        except RouterError:
            return "routine"
        for label in ("trivial", "routine", "hard"):
            if re.search(rf"\b{label}\b", out):
                return label
        return "routine"

    def tier_call(self, prompt, *, complexity: str = "auto", system: Optional[str] = None,
                  max_tokens: int = 1024, accept: Optional[Callable[[str], bool]] = None,
                  **kw) -> Completion:
        """"Small model does the menial work, escalate on hard."

        ``complexity``: ``trivial`` | ``routine`` | ``hard`` | ``critical`` | ``auto``.
        Each maps to a starting tier via ``complexity_tiers`` in the config (default:
        fast / medium / sonnet / opus); the chain's fallbacks provide further
        escalation, and ``accept=`` can force a step up when an answer is not good enough.
        """
        if complexity == "auto":
            complexity = self.classify_complexity(prompt if isinstance(prompt, str) else json.dumps(prompt))
        tiers = self.config.complexity_tiers
        name = tiers.get(complexity) or tiers.get("routine") or self.config.role("medium")
        return self.complete(prompt, system=system, tier=name, max_tokens=max_tokens,
                             accept=accept, task=f"tier_call:{complexity}", **kw)


# --------------------------------------------------------------------------- #
# Module-level convenience (lazy default router)
# --------------------------------------------------------------------------- #

_DEFAULT: Optional[Router] = None


def get_router(reload: bool = False) -> Router:
    """The process-wide default router, built from the config file + environment."""
    global _DEFAULT
    if _DEFAULT is None or reload:
        _DEFAULT = Router()
    return _DEFAULT


def set_router(router: Optional[Router]) -> None:
    global _DEFAULT
    _DEFAULT = router

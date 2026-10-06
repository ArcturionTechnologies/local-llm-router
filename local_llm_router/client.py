"""HTTP layer: one function that can talk to any configured tier.

Speaks two wire formats:

* ``openai``    -- ``POST {base_url}/chat/completions`` (LM Studio, ``mlx_lm.server``,
  llama.cpp, Ollama's OpenAI endpoint, Groq, Cerebras, Gemini's OpenAI endpoint...).
* ``anthropic`` -- ``POST {base_url}/v1/messages``.

Standard library only. The transport is injectable so tests (and callers with
their own HTTP stack) can swap it out.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from .config import Tier
from .errors import MissingCredentials, RateLimited, TierBusy, TierError

ANTHROPIC_VERSION = "2023-06-01"

#: ``(method, url, json_body_or_None, headers, timeout) -> (status, body_bytes)``.
#: Must raise :class:`TierError` when the server cannot be reached at all, and
#: must *return* (not raise) HTTP error statuses.
Transport = Callable[[str, str, Optional[dict], dict, float], tuple]


def urllib_transport(method: str, url: str, payload: Optional[dict],
                     headers: dict, timeout: float) -> tuple:
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:  # noqa: S310 - URL comes from config
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        raise TierError(f"unreachable: {e}") from e


@dataclass
class Usage:
    tokens_in: int = 0
    tokens_out: int = 0
    real: bool = False        # True when the server reported the counts


def estimate_tokens(value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, default=str)
    return max(1, len(text) // 4)


def usage_counts(data) -> Usage:
    """Real token counts from OpenAI-, Anthropic- or Ollama-style usage blocks."""
    if not isinstance(data, dict):
        return Usage()
    u = data.get("usage") if isinstance(data.get("usage"), dict) else {}
    tin = u.get("prompt_tokens") or u.get("input_tokens") or data.get("prompt_eval_count") or 0
    tout = u.get("completion_tokens") or u.get("output_tokens") or data.get("eval_count") or 0
    try:
        tin, tout = max(0, int(tin)), max(0, int(tout))
    except (TypeError, ValueError):
        return Usage()
    return Usage(tin, tout, real=bool(tin or tout))


def busy_from_body(body) -> Optional[TierBusy]:
    """Return :class:`TierBusy` iff ``body`` is the ``local-pool-busy`` contract."""
    try:
        data = json.loads(body)
    except Exception:
        return None
    if not isinstance(data, dict) or data.get("error") != "local-pool-busy":
        return None
    holder, pid = data.get("holder"), data.get("holder_pid")
    return TierBusy(f"local-pool-busy -- held by {holder} (pid {pid}); fall through",
                    holder=holder, holder_pid=pid)


def _extract_openai(data) -> str:
    try:
        msg = data["choices"][0]["message"]
        return (msg.get("content") or msg.get("reasoning") or "").strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise TierError(f"malformed completion response: {str(data)[:200]}") from None


def _extract_anthropic(data) -> str:
    try:
        parts = [b.get("text", "") for b in data["content"] if b.get("type", "text") == "text"]
        return "".join(parts).strip()
    except (KeyError, TypeError, AttributeError):
        raise TierError(f"malformed completion response: {str(data)[:200]}") from None


def build_request(tier: Tier, messages: list, max_tokens: int, temperature: float,
                  env: Optional[Mapping[str, str]] = None) -> tuple:
    """Return ``(url, payload, headers)`` for a tier. Raises if a cloud key is missing."""
    env = os.environ if env is None else env
    key = env.get(tier.api_key_env, "") if tier.api_key_env else ""
    if tier.kind == "cloud" and tier.api_key_env and not key:
        raise MissingCredentials(f"{tier.name}: ${tier.api_key_env} is not set")
    base = tier.base_url.rstrip("/")

    if tier.protocol == "anthropic":
        system = "\n".join(m["content"] for m in messages
                           if m["role"] == "system" and isinstance(m["content"], str))
        convo = [m for m in messages if m["role"] != "system"]
        payload = {"model": tier.model, "max_tokens": max_tokens,
                   "temperature": temperature, "messages": convo}
        if system:
            payload["system"] = system
        headers = {"anthropic-version": ANTHROPIC_VERSION}
        if key:
            headers["x-api-key"] = key
        url = f"{base}/v1/messages"
    else:
        payload = {"model": tier.model, "messages": messages,
                   "max_tokens": max_tokens, "temperature": temperature}
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        url = f"{base}/chat/completions"
    payload.update(tier.extra_payload)
    return url, payload, headers


def complete(tier: Tier, messages: list, *, max_tokens: int = 400, temperature: float = 0.2,
             timeout: Optional[float] = None, transport: Transport = urllib_transport,
             env: Optional[Mapping[str, str]] = None) -> tuple:
    """One request to one tier. Returns ``(text, Usage)``.

    Raises :class:`TierBusy` (lock contention), :class:`RateLimited` (429/402),
    :class:`MissingCredentials`, or :class:`TierError` (everything else).
    """
    url, payload, headers = build_request(tier, messages, max_tokens, temperature, env)
    status, body = transport("POST", url, payload, headers, timeout or tier.timeout)
    if status == 503:
        busy = busy_from_body(body)
        if busy is not None:
            raise busy
    if status in (429, 402):
        raise RateLimited(f"{tier.name}: HTTP {status}")
    if not 200 <= status < 300:
        raise TierError(f"{tier.name}: HTTP {status}: {body[:200]!r}")
    try:
        data = json.loads(body)
    except ValueError as e:
        raise TierError(f"{tier.name}: response is not JSON: {body[:200]!r}") from e
    # Belt and braces: a server may answer the busy contract with a 200.
    if isinstance(data, dict) and data.get("error") == "local-pool-busy":
        raise TierBusy(f"local-pool-busy -- held by {data.get('holder')}; fall through",
                       holder=data.get("holder"), holder_pid=data.get("holder_pid"))
    text = _extract_anthropic(data) if tier.protocol == "anthropic" else _extract_openai(data)
    return text, usage_counts(data)


def health(tier: Tier, *, timeout: float = 1.5, transport: Transport = urllib_transport,
           env: Optional[Mapping[str, str]] = None) -> bool:
    """Cheap liveness probe. Local: ``GET {base_url}/models`` answers 2xx.
    Cloud: true when the API key is present (no network call, no cost)."""
    env = os.environ if env is None else env
    if tier.kind == "cloud":
        return bool(env.get(tier.api_key_env)) if tier.api_key_env else True
    try:
        status, _ = transport("GET", tier.base_url.rstrip("/") + "/models", None, {}, timeout)
    except TierError:
        return False
    return 200 <= status < 300


def image_part(path) -> dict:
    """An OpenAI-vision ``image_url`` content part with the file inlined as a data URL."""
    rp = Path(path).expanduser().resolve()
    if not rp.is_file():
        raise TierError(f"image path not found: {path}")
    mime = mimetypes.guess_type(rp.name)[0] or "application/octet-stream"
    b64 = base64.b64encode(rp.read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}}

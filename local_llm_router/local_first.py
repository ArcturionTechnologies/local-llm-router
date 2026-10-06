"""Local-first policy gate: warn once, then block, routine work sent to paid APIs.

Routing tables only help if callers use them. This gate is the enforcement
side: ask it before a paid call and it tells you whether the work should have
stayed on a local or free tier.

    decision, reason = LocalFirstGate(config).gate("claude-sonnet", task_class="classify")

Rules, in order:

1. Not a paid provider                      -> ``allow``
2. Task class in ``exempt_task_classes``     -> ``allow``  (money, external sends, prod deploys...)
3. Stakes ``high`` or ``irreversible``       -> ``allow``
4. Complexity ``complex``                    -> ``allow``
5. Otherwise: first offence in this session  -> ``warn``; every later one -> ``block``

"Session" is ``$CLAUDE_SESSION_ID`` / ``$SESSION_ID`` when set, else the UTC day.

It can also run as a Claude Code ``PreToolUse`` hook (reads the tool-call JSON on
stdin, prints a decision JSON): it spots shell commands that invoke a paid CLI
or SDK (``claude -p``, ``codex -p``, ``anthropic.messages.create``...) and applies
the same rules.

    llm-router local-first hook      # PreToolUse hook entry
    llm-router local-first test      # print the decision matrix
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Optional

from .config import Config


class LocalFirstGate:
    def __init__(self, config: Config, env: Optional[Mapping[str, str]] = None):
        self.cfg = config.local_first
        self.path = config.state_path / "local-first-violations.json"
        self._env = os.environ if env is None else env
        # Any tier configured as paid also counts as a paid provider.
        self.paid = {p.lower() for p in self.cfg.paid_providers}
        self.paid |= {t.name.lower() for t in config.tiers.values() if t.cost_class == "paid"}

    # -- state --------------------------------------------------------------
    def session_key(self) -> str:
        sid = self._env.get("CLAUDE_SESSION_ID") or self._env.get("SESSION_ID")
        return sid or datetime.now(timezone.utc).strftime("daily-%Y-%m-%d")

    def load(self) -> dict:
        try:
            return json.loads(self.path.read_text())
        except (OSError, ValueError):
            return {}

    def _save(self, data: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))

    def _bump(self, key: str) -> int:
        data = self.load()
        bucket = data.setdefault(key, {"count": 0, "first_at": None, "last_at": None})
        bucket["count"] = bucket.get("count", 0) + 1
        now = datetime.now(timezone.utc).isoformat(timespec="seconds")
        bucket["first_at"] = bucket.get("first_at") or now
        bucket["last_at"] = now
        self._save(data)
        return bucket["count"]

    def clear(self) -> None:
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass

    # -- decision -----------------------------------------------------------
    def is_paid(self, provider: str) -> bool:
        return (provider or "").lower() in self.paid

    def gate(self, provider: str, task_class: str = "other", complexity: str = "simple",
             stakes: str = "low") -> tuple:
        """Return ``("allow"|"warn"|"block", reason_or_None)``."""
        if not self.is_paid(provider):
            return "allow", None
        if (task_class or "").lower() in {c.lower() for c in self.cfg.exempt_task_classes}:
            return "allow", f"task-class={task_class} exempt"
        if (stakes or "").lower() in ("irreversible", "high"):
            return "allow", f"stakes={stakes}"
        if (complexity or "").lower() == "complex":
            return "allow", "complexity=complex"
        count = self._bump(self.session_key())
        reason = (f"paid={provider}, task={task_class}, complexity={complexity}, "
                  f"stakes={stakes} -- should have used a local or free tier")
        if count == 1:
            return "warn", f"first offense this session -- {reason}"
        return "block", f"violation #{count} this session -- {reason}"

    def hook(self, stdin=None, stdout=None) -> dict:
        """PreToolUse hook: tool-call JSON in, decision JSON out."""
        stdin, stdout = stdin or sys.stdin, stdout or sys.stdout

        def emit(obj: dict) -> dict:
            stdout.write(json.dumps(obj) + "\n")
            return obj

        try:
            event = json.load(stdin)
        except ValueError:
            return emit({"decision": "allow"})
        tool = event.get("tool_name") or event.get("tool", "")
        cmd = (event.get("input") or {}).get("command", "") or event.get("command", "")
        if tool in self.cfg.shell_tools and any(m in cmd for m in self.cfg.paid_markers):
            provider = "claude" if "claude" in cmd else "codex"
            decision, reason = self.gate(provider, task_class="other", complexity="simple", stakes="low")
            if decision == "block":
                return emit({"decision": "deny",
                             "reason": f"local-first policy: {reason}. Use pick_tier() and a local tier first."})
            if decision == "warn":
                return emit({"decision": "allow", "message": f"local-first policy warning: {reason}"})
        return emit({"decision": "allow"})

    def status(self) -> str:
        data = self.load()
        if not data:
            return "No local-first violations recorded."
        return "\n".join(f"  {k}: {b['count']} violations  ({b.get('first_at', '?')} -> {b.get('last_at', '?')})"
                         for k, b in sorted(data.items()))

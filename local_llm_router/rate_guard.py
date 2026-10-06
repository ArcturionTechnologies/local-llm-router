"""Session rate guard: don't spawn more paid-agent sessions than your plan allows.

If you launch headless agent sessions (for example ``claude -p ...`` from cron,
a dispatcher or a file watcher), subscription rate limits are the next wall
after local capacity. The guard is called *before every spawn* and answers with
a status and an exit code:

* ``ok`` (exit 0) -- go ahead
* ``throttle`` (exit 1) -- approaching a ceiling or in cool-down; defer low-priority work
* ``block`` (exit 2) -- hard stop: kill switch, concurrency or rate ceiling hit

Signals:

* concurrent sessions -- ``pgrep -f <process_pattern>``
* rolling spawn counts (1 h / 5 h / 7 d) -- from a JSONL manifest that
  callers append to with :meth:`RateGuard.log_spawn`
* explicit rate-limit text (``429``, ``overloaded``...) in recent spawn logs ->
  a cool-down window
* a kill switch: create ``<state_dir>/KILL_ACTIVE`` (or any file in
  ``rate_guard.kill_flags``) to block everything

    python -m local_llm_router guard check ; echo $?
"""

from __future__ import annotations

import json
import re
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import Config


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class RateGuard:
    def __init__(self, config: Config, *,
                 now: Callable[[], datetime] = _utcnow,
                 active_counter: Optional[Callable[[], int]] = None,
                 run=subprocess.run):
        self.cfg = config.rate_guard
        self.dir = config.state_path / "rate-guard"
        self.manifest = self.dir / "spawns.jsonl"
        self.cooldown_file = self.dir / "cooldown.json"
        self.state_file = self.dir / "state.json"
        self.kill_flags = [config.state_path / "KILL_ACTIVE",
                           *[Path(p).expanduser() for p in self.cfg.kill_flags]]
        self._now = now
        self._run = run
        self._active_counter = active_counter

    # -- signals ------------------------------------------------------------
    def now_iso(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def count_active(self) -> int:
        if self._active_counter:
            return self._active_counter()
        if not self.cfg.process_pattern:
            return 0
        try:
            r = self._run(["pgrep", "-f", self.cfg.process_pattern],
                          capture_output=True, text=True, check=False)
            return len([p for p in r.stdout.splitlines() if p.strip()])
        except (OSError, subprocess.SubprocessError):
            return 0

    def count_spawns(self, hours: float, scan_lines: int = 5000) -> int:
        """Count ``spawn_fired`` / ``session_start`` events in the last ``hours``."""
        cutoff = (self._now() - timedelta(hours=hours)).isoformat(timespec="seconds")
        sources = [self.manifest]
        if self.cfg.registry_path:
            sources.append(Path(self.cfg.registry_path).expanduser())
        count = 0
        for src in sources:
            try:
                lines = src.read_text().splitlines()[-scan_lines:]
            except OSError:
                continue
            for line in lines:
                try:
                    d = json.loads(line)
                except ValueError:
                    continue
                if d.get("ts", "") >= cutoff and d.get("event") in ("spawn_fired", "session_start"):
                    count += 1
        return count

    def in_cooldown(self) -> tuple:
        try:
            d = json.loads(self.cooldown_file.read_text())
            until = datetime.fromisoformat(d["until"])
        except (OSError, ValueError, KeyError):
            return False, 0
        now = self._now()
        if until > now:
            return True, int((until - now).total_seconds() / 60) + 1
        return False, 0

    def set_cooldown(self, reason: str = "rate_limit_signal") -> None:
        until = self._now() + timedelta(minutes=self.cfg.cooldown_min)
        self.cooldown_file.parent.mkdir(parents=True, exist_ok=True)
        self.cooldown_file.write_text(json.dumps({
            "until": until.isoformat(timespec="seconds"),
            "reason": reason, "set_at": self.now_iso()}))

    def scan_logs_for_rate_limit(self) -> bool:
        if not self.cfg.spawn_log_dir:
            return False
        d = Path(self.cfg.spawn_log_dir).expanduser()
        pattern = re.compile(self.cfg.rate_limit_regex, re.IGNORECASE)
        try:
            logs = sorted(d.glob("*.log"), key=lambda f: f.stat().st_mtime, reverse=True)[:5]
        except OSError:
            return False
        for lf in logs:
            try:
                if pattern.search(lf.read_text()[-2000:]):
                    return True
            except OSError:
                continue
        return False

    # -- decision -----------------------------------------------------------
    def _write_state(self, record: dict) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(record, indent=2))
        except OSError:
            pass

    def check(self) -> dict:
        c = self.cfg
        kill = next((f for f in self.kill_flags if f.exists()), None)
        base = {"ts": self.now_iso(), "active_max": c.max_concurrent,
                "spawns_1h_max": c.max_1h, "spawns_5h_max": c.max_5h,
                "spawns_week_max": c.max_week}
        if kill:
            rec = {**base, "status": "block", "active": 0, "spawns_1h": 0, "spawns_5h": 0,
                   "spawns_week": 0, "peak_util": 0.0, "cooldown": False,
                   "cooldown_remaining_min": 0,
                   "reason": f"KILL SWITCH ACTIVE at {kill} -- all spawns blocked"}
            self._write_state(rec)
            return rec

        active = self.count_active()
        s1, s5, sw = self.count_spawns(1), self.count_spawns(5), self.count_spawns(168, 20000)
        cool, remaining = self.in_cooldown()
        util = {"1h": s1 / max(c.max_1h, 1), "5h": s5 / max(c.max_5h, 1),
                "week": sw / max(c.max_week, 1)}
        peak = max(util.values())
        rec = {**base, "status": "ok", "active": active, "spawns_1h": s1, "spawns_5h": s5,
               "spawns_week": sw, "peak_util": round(peak, 3), "cooldown": cool,
               "cooldown_remaining_min": remaining, "reason": ""}

        if cool:
            rec.update(status="throttle", reason=f"rate-limit cooldown -- {remaining}min remaining")
        elif active >= c.max_concurrent:
            rec.update(status="block", reason=f"max concurrent sessions ({c.max_concurrent}) reached")
        elif peak >= c.red_pct:
            rec.update(status="block", reason=(
                f"rate ceiling hit (1h={util['1h']:.0%}, 5h={util['5h']:.0%}, week={util['week']:.0%})"))
        elif peak >= c.yellow_pct:
            rec.update(status="throttle", reason=f"approaching ceiling (peak={peak:.0%}) -- defer low-priority")
        elif self.scan_logs_for_rate_limit():
            self.set_cooldown("rate_limit_signal_in_logs")
            rec.update(status="throttle", cooldown=True, cooldown_remaining_min=c.cooldown_min,
                       reason=f"rate-limit signal detected in spawn logs -- {c.cooldown_min}min cooldown set")
        self._write_state(rec)
        return rec

    def log_spawn(self, caller: str, task_id: str, status: str = "spawn_fired") -> None:
        """Append a spawn event. Call this right after you launch a session."""
        self.manifest.parent.mkdir(parents=True, exist_ok=True)
        with self.manifest.open("a") as f:
            f.write(json.dumps({"ts": self.now_iso(), "event": status,
                                "caller": caller, "task_id": task_id}) + "\n")

    @staticmethod
    def exit_code(record: dict) -> int:
        return {"ok": 0, "throttle": 1, "block": 2}[record["status"]]

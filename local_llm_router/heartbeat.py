"""Heartbeat gate: skip scheduled ticks when nothing has changed -- or the host is busy.

A cron/launchd "heartbeat" that wakes an LLM every few minutes burns tokens
(and fans) even when there is nothing to do. Each heartbeat declares what it
cares about in a *signals manifest*; on every tick the gate fingerprints those
signals and fires only if something changed.

Manifest -- ``<home>/heartbeat/signals/<heartbeat_id>.json``::

    {
      "files":    ["~/notes/inbox.md"],
      "dirs":     ["~/notes/approvals/"],
      "commands": [{"name": "queue_size", "cmd": "wc -l ~/notes/queue.md"}],
      "sanity_fire_hours": 6,
      "skip_when_hot": true,
      "log_path": "~/.config/local-llm-router/state/heartbeat/inbox.log"
    }

Decision per tick:

* fingerprint changed                       -> **fire**
* unchanged, but ``sanity_fire_hours`` passed -> **fire** (catches silent failures)
* otherwise                                 -> **skip**
* no manifest                               -> **skip** (fail closed: no declared scope)
* ``skip_when_hot`` and the thermal gate is red -> **skip** (back off; state is not
  advanced, so the change is picked up on the next cool tick)

Manifest ``commands`` run through the shell: manifests are trusted local config.

CLI / shell integration::

    llm-router heartbeat should-fire inbox || exit 0   # exit 0 = fire, 1 = skip
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import Config
from .thermal import ThermalGate


def _expand(p: str) -> Path:
    return Path(os.path.expanduser(p))


def _hash_file(path: Path) -> str:
    """mtime + size: cheap, catches edits without reading the content."""
    try:
        st = path.stat()
        return f"{int(st.st_mtime)}:{st.st_size}"
    except FileNotFoundError:
        return "MISSING"
    except OSError as e:
        return f"ERR:{type(e).__name__}"


def _hash_dir(path: Path) -> str:
    """Hash of a directory's children (one level): name, mtime, size."""
    try:
        if not path.is_dir():
            return "NOT_A_DIR"
        entries = []
        for child in sorted(path.iterdir()):
            try:
                st = child.stat()
                entries.append(f"{child.name}:{int(st.st_mtime)}:{st.st_size}")
            except OSError:
                entries.append(f"{child.name}:ERR")
        return hashlib.sha256("|".join(entries).encode()).hexdigest()[:16]
    except OSError as e:
        return f"ERR:{type(e).__name__}"


def _hash_command(cmd: str, timeout: int = 5) -> str:
    try:
        r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)  # noqa: S602
        return hashlib.sha256(r.stdout.encode()).hexdigest()[:16]
    except (OSError, subprocess.SubprocessError) as e:
        return f"ERR:{type(e).__name__}"


def compute_fingerprint(manifest: dict) -> str:
    parts = []
    for f in manifest.get("files", []):
        parts.append(f"file:{f}={_hash_file(_expand(f))}")
    for d in manifest.get("dirs", []):
        parts.append(f"dir:{d}={_hash_dir(_expand(d))}")
    for c in manifest.get("commands", []):
        cmd = c.get("cmd", "") if isinstance(c, dict) else str(c)
        name = c.get("name", "cmd") if isinstance(c, dict) else "cmd"
        parts.append(f"cmd:{name}={_hash_command(cmd)}")
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()[:24]


class HeartbeatGate:
    def __init__(self, config: Config, *, thermal: Optional[ThermalGate] = None,
                 now: Callable[[], datetime] = lambda: datetime.now(timezone.utc)):
        hb = config.heartbeat
        self.signals_dir = _expand(hb.signals_dir) if hb.signals_dir else config.home / "heartbeat" / "signals"
        self.state_dir = _expand(hb.state_dir) if hb.state_dir else config.state_path / "heartbeat"
        self.tasks_dir = _expand(hb.tasks_dir) if hb.tasks_dir else None
        self.default_sanity_hours = hb.sanity_fire_hours
        self.thermal = thermal or ThermalGate(config.thermal)
        self._now = now

    # -- io -----------------------------------------------------------------
    def _now_iso(self) -> str:
        return self._now().isoformat(timespec="seconds")

    def load_manifest(self, hbid: str) -> Optional[dict]:
        try:
            return json.loads((self.signals_dir / f"{hbid}.json").read_text())
        except (OSError, ValueError):
            return None

    def load_state(self, hbid: str) -> dict:
        try:
            return json.loads((self.state_dir / f"{hbid}.json").read_text())
        except (OSError, ValueError):
            return {}

    def save_state(self, hbid: str, state: dict) -> None:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        (self.state_dir / f"{hbid}.json").write_text(json.dumps(state, indent=2))

    def _log(self, manifest: dict, line: str) -> None:
        if not manifest.get("log_path"):
            return
        p = _expand(manifest["log_path"])
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with p.open("a") as f:
                f.write(line.rstrip() + "\n")
        except OSError:
            pass                               # logging must never break the gate

    # -- decision -----------------------------------------------------------
    def should_fire(self, hbid: str) -> bool:
        """True if the heartbeat should run this tick.

        Side effects: firing records the new fingerprint and time; skipping
        bumps a skip counter for observability.
        """
        manifest = self.load_manifest(hbid)
        if manifest is None:
            return False                       # no declared scope: fail closed

        state = self.load_state(hbid)
        if manifest.get("skip_when_hot") and self.thermal.check("light").action == "block":
            state["last_skip_at"] = self._now_iso()
            state["last_skip_reason"] = "host_hot"
            self.save_state(hbid, state)
            self._log(manifest, f"[{self._now_iso()}] gated: host busy/hot")
            return False

        fingerprint = compute_fingerprint(manifest)
        sanity_hours = manifest.get("sanity_fire_hours", self.default_sanity_hours)
        last_fp, last_fire = state.get("last_fingerprint"), state.get("last_fire_at")

        sanity_due = True
        if last_fire:
            try:
                sanity_due = (self._now() - datetime.fromisoformat(last_fire)) >= timedelta(hours=sanity_hours)
            except ValueError:
                sanity_due = True
        changed = last_fp != fingerprint

        if changed or sanity_due or not last_fire:
            reason = ("first_fire" if not last_fire
                      else "fingerprint_changed" if changed else "sanity_floor")
            state.update(last_fingerprint=fingerprint, last_fire_at=self._now_iso(),
                         last_fire_reason=reason, skip_count_since_last_fire=0)
            self.save_state(hbid, state)
            self._log(manifest, f"[{self._now_iso()}] firing: {reason}")
            return True

        skips = state.get("skip_count_since_last_fire", 0) + 1
        state.update(skip_count_since_last_fire=skips, last_skip_at=self._now_iso(),
                     last_fingerprint_seen=fingerprint)
        self.save_state(hbid, state)
        self._log(manifest, f"[{self._now_iso()}] gated: signals unchanged (skip #{skips})")
        return False

    def status(self, hbid: str) -> dict:
        manifest = self.load_manifest(hbid)
        return {"hbid": hbid, "manifest_present": manifest is not None, "manifest": manifest,
                "state": self.load_state(hbid),
                "current_fingerprint": compute_fingerprint(manifest) if manifest else None}

    def reset(self, hbid: str) -> None:
        """Force the next tick to fire."""
        state = self.load_state(hbid)
        state.pop("last_fingerprint", None)
        state.pop("last_fire_at", None)
        self.save_state(hbid, state)

    def discover(self) -> list:
        """List heartbeat ids found under ``heartbeat.tasks_dir`` and whether each has a manifest."""
        if not self.tasks_dir or not self.tasks_dir.exists():
            return []
        return [{"hbid": d.name, "has_manifest": (self.signals_dir / f"{d.name}.json").exists()}
                for d in sorted(self.tasks_dir.iterdir()) if d.is_dir()]

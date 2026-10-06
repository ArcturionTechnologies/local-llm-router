"""Thermal / load gate: back off local inference when the Mac runs hot or busy.

A local model is only worth running when the machine has headroom. The gate
reads four cheap signals and classifies the host:

=======  ==========================================================
red      CPU load > 80% of cores, free memory < 8%, battery < 15%
         (unplugged) or macOS is throttling the CPU (speed limit < 90).
         Every local tier is blocked.
yellow   CPU load > 50% of cores, or free memory < 15%.
unknown  The load or thermal sensor could not be read. Never treated as green.
green    Everything else.
=======  ==========================================================

Each tier carries a *profile* that decides what yellow/unknown mean for it:

========  ===================================================================
light     Small model. Only red blocks.
standard  Mid-size model. Red blocks; unreadable sensors block (conservative).
medium    Larger model. Yellow or unknown sensors -> ``downgrade`` to a lighter tier.
coder     Code model. Yellow or unknown sensors -> blocked (fall through).
vision    Vision model. Yellow or unknown sensors -> blocked (fall through).
heavy     Big MoE. Yellow *or* unknown sensors -> blocked. Never runs under pressure.
none      Not gated (cloud tiers).
========  ===================================================================

Instead of the built-in probe you can point ``thermal.command`` at your own
script. It is called as ``<command> <profile>`` and follows the same contract:
exit 0 = go, exit 1 = lighter tier please (``medium`` only, otherwise blocked),
exit 2 = blocked. A missing script fails open; a script that hangs fails closed.

Set ``LLM_ROUTER_BYPASS_THERMAL=1`` for lean cron/heartbeat jobs that should
never be gated.
"""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping, Optional

from .config import ThermalConfig
from .errors import ThermalBlocked, ThermalDowngrade

Run = Callable[..., "subprocess.CompletedProcess"]


@dataclass
class Snapshot:
    """Raw host readings. ``None`` means the sensor could not be read."""

    cpu_load: Optional[float] = None       # 1-minute load average
    cpu_count: int = 1
    mem_free_pct: Optional[float] = None   # system-wide free memory, percent
    on_battery: bool = False
    battery_pct: Optional[int] = None
    speed_limit: Optional[int] = None      # pmset CPU_Speed_Limit; None = not throttling
    thermal_known: bool = True             # False when the thermal sensor is unreadable


@dataclass
class Decision:
    action: str                            # "allow" | "downgrade" | "block"
    level: str                             # "green" | "yellow" | "red" | "unknown"
    reason: str = ""

    @property
    def exit_code(self) -> int:
        return {"allow": 0, "downgrade": 1, "block": 2}[self.action]


# --------------------------------------------------------------------------- #
# Probing
# --------------------------------------------------------------------------- #

def _out(run: Run, cmd: list, timeout: float = 3.0) -> str:
    try:
        r = run(cmd, capture_output=True, text=True, timeout=timeout, check=False)
        return r.stdout or ""
    except (OSError, subprocess.SubprocessError):
        return ""


def _free_pct_from_vm_stat(text: str) -> Optional[float]:
    pages = {}
    for key in ("free", "active", "inactive", "wired down"):
        m = re.search(rf"Pages {key}:\s+(\d+)", text)
        if m:
            pages[key] = int(m.group(1))
    if "free" not in pages:
        return None
    total = sum(pages.values()) + 1
    return pages["free"] * 100.0 / total


def probe(run: Run = subprocess.run, platform: str = sys.platform) -> Snapshot:
    """Read the host. Uses macOS tools on Darwin, stdlib/procfs elsewhere."""
    snap = Snapshot(cpu_count=os.cpu_count() or 1)
    try:
        snap.cpu_load = os.getloadavg()[0]
    except (OSError, AttributeError):
        snap.cpu_load = None

    if platform == "darwin":
        mp = _out(run, ["memory_pressure"])
        m = re.search(r"System-wide memory free percentage:\s*(\d+)%", mp)
        if m:
            snap.mem_free_pct = float(m.group(1))
        else:
            snap.mem_free_pct = _free_pct_from_vm_stat(_out(run, ["vm_stat"]))

        ps = _out(run, ["pmset", "-g", "ps"])
        snap.on_battery = "Battery Power" in ps.splitlines()[0] if ps.strip() else False
        bm = re.search(r"(\d+)%", _out(run, ["pmset", "-g", "batt"]))
        snap.battery_pct = int(bm.group(1)) if bm else None

        therm = _out(run, ["pmset", "-g", "therm"])
        if not therm.strip():
            snap.thermal_known = False
        else:
            sm = re.search(r"CPU_Speed_Limit\s*=?\s*(\S+)", therm)
            if sm is None:
                snap.speed_limit = None               # not throttling
            elif sm.group(1).isdigit():
                snap.speed_limit = int(sm.group(1))
            else:
                snap.thermal_known = False
    else:  # Linux and friends: no thermal API, assume no throttling
        try:
            info = Path("/proc/meminfo").read_text()
            total = int(re.search(r"MemTotal:\s+(\d+)", info).group(1))
            avail = int(re.search(r"MemAvailable:\s+(\d+)", info).group(1))
            snap.mem_free_pct = avail * 100.0 / total
        except (OSError, AttributeError, ValueError):
            snap.mem_free_pct = None
    return snap


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #

def classify(snap: Snapshot, cfg: ThermalConfig) -> tuple:
    """Return ``(level, flags, reason)`` for a snapshot."""
    cpu_unknown = snap.cpu_load is None
    cpus = max(snap.cpu_count, 1)
    cpu_heavy = (not cpu_unknown) and snap.cpu_load > cfg.cpu_heavy_ratio * cpus
    cpu_mod = (not cpu_unknown) and snap.cpu_load > cfg.cpu_moderate_ratio * cpus
    mem_critical = snap.mem_free_pct is not None and snap.mem_free_pct < cfg.mem_critical_pct
    mem_low = snap.mem_free_pct is not None and snap.mem_free_pct < cfg.mem_low_pct
    batt_low = bool(snap.on_battery and snap.battery_pct is not None
                    and snap.battery_pct < cfg.battery_low_pct)
    throttled = snap.speed_limit is not None and snap.speed_limit < cfg.throttle_speed_limit
    unknown = cpu_unknown or not snap.thermal_known

    flags = {"cpu_heavy": cpu_heavy, "cpu_mod": cpu_mod, "mem_critical": mem_critical,
             "mem_low": mem_low, "batt_low": batt_low, "throttled": throttled, "unknown": unknown}
    detail = (f"cpu_load={snap.cpu_load if snap.cpu_load is None else round(snap.cpu_load, 2)}/{cpus} "
              f"mem_free={snap.mem_free_pct if snap.mem_free_pct is None else round(snap.mem_free_pct)}% "
              f"batt={snap.battery_pct}% throttled={int(throttled)}")
    if cpu_heavy or mem_critical or batt_low or throttled:
        return "red", flags, detail
    if cpu_mod or mem_low:
        return "yellow", flags, detail
    if unknown:
        return "unknown", flags, detail
    return "green", flags, detail


def decide(snap: Snapshot, profile: str, cfg: ThermalConfig) -> Decision:
    """Apply the profile rules to a snapshot."""
    if profile == "none":
        return Decision("allow", "green", "ungated tier")
    level, flags, detail = classify(snap, cfg)
    if level == "red":
        return Decision("block", "red", f"SKIP_LLM: {detail}")
    if profile == "heavy" and level in ("yellow", "unknown"):
        return Decision("block", level, f"HEAVY_BLOCK: {level} host state ({detail}) -- heavy model must not run under pressure")
    if profile == "light":
        return Decision("allow", level, detail)
    if flags["unknown"]:
        reason = f"DEGRADED: load/thermal sensor unreadable ({detail}) -- capping at a light model"
        return Decision("downgrade" if profile == "medium" else "block", "unknown", reason)
    if level == "yellow" and profile in ("medium", "coder", "vision"):
        action = "downgrade" if profile == "medium" else "block"
        return Decision(action, "yellow", f"{profile.upper()}_YELLOW: {detail}")
    return Decision("allow", level, detail)


# --------------------------------------------------------------------------- #
# RAM-pressure demotion for heavy tiers
# --------------------------------------------------------------------------- #

def free_memory_gb(run: Run = subprocess.run, platform: str = sys.platform) -> Optional[float]:
    """Free + reclaimable memory in GiB, or ``None`` if it cannot be read."""
    if platform == "darwin":
        text = _out(run, ["vm_stat"], timeout=2.0)
        size = re.search(r"page size of (\d+) bytes", text)
        free = re.search(r"Pages free:\s+(\d+)", text)
        inactive = re.search(r"Pages inactive:\s+(\d+)", text)
        if size and free:
            pages = int(free.group(1)) + (int(inactive.group(1)) if inactive else 0)
            return pages * int(size.group(1)) / (1024 ** 3)
        return None
    try:
        info = Path("/proc/meminfo").read_text()
        return int(re.search(r"MemAvailable:\s+(\d+)", info).group(1)) / (1024 ** 2)
    except (OSError, AttributeError, ValueError):
        return None


def should_demote_heavy(cfg: ThermalConfig, run: Run = subprocess.run,
                        platform: str = sys.platform) -> bool:
    """True when a heavy tier should step down to a lighter one.

    Triggers: another heavy session is running (``pressure_process_pattern``
    matches a process that is not in ``pressure_ignore``), or free memory is
    below ``min_free_gb_heavy``.
    """
    if cfg.pressure_process_pattern:
        out = _out(run, ["pgrep", "-fl", cfg.pressure_process_pattern], timeout=2.0)
        for line in out.splitlines():
            if not line.strip():
                continue
            if any(skip in line for skip in cfg.pressure_ignore):
                continue
            return True
    free = free_memory_gb(run, platform)
    return free is not None and free < cfg.min_free_gb_heavy


# --------------------------------------------------------------------------- #
# Gate
# --------------------------------------------------------------------------- #

class ThermalGate:
    """Decides whether a tier may run right now."""

    def __init__(self, cfg: ThermalConfig, *,
                 prober: Optional[Callable[[], Snapshot]] = None,
                 run: Run = subprocess.run,
                 env: Optional[Mapping[str, str]] = None):
        self.cfg = cfg
        self._prober = prober or (lambda: probe(run))
        self._run = run
        self._env = os.environ if env is None else env

    def check(self, profile: str) -> Decision:
        if not self.cfg.enabled or profile == "none":
            return Decision("allow", "green", "gate disabled" if not self.cfg.enabled else "ungated tier")
        if self._env.get(self.cfg.bypass_env) == "1":
            return Decision("allow", "green", f"{self.cfg.bypass_env}=1")
        if self.cfg.command:
            return self._check_command(profile)
        return decide(self._prober(), profile, self.cfg)

    def enforce(self, profile: str) -> None:
        """Raise :class:`ThermalBlocked` / :class:`ThermalDowngrade` unless allowed."""
        d = self.check(profile)
        if d.action == "downgrade":
            raise ThermalDowngrade(d.reason or "thermal gate requested a lighter tier")
        if d.action == "block":
            raise ThermalBlocked(d.reason or f"thermal gate blocked profile {profile!r}")

    def should_demote(self) -> bool:
        if not self.cfg.enabled or self._env.get(self.cfg.bypass_env) == "1":
            return False
        return should_demote_heavy(self.cfg, self._run)

    # -- external command contract -----------------------------------------
    def _check_command(self, profile: str) -> Decision:
        argv = shlex.split(self.cfg.command)
        exe = Path(argv[0]).expanduser()
        if exe.is_absolute() and not exe.exists():
            return Decision("allow", "unknown", f"gate command missing: {exe} (fail open)")
        try:
            r = self._run([*argv, profile], text=True, capture_output=True, timeout=3, check=False)
        except subprocess.TimeoutExpired:
            return Decision("block", "unknown", "thermal gate timed out; skip local LLM")
        except OSError:
            return Decision("allow", "unknown", "gate command not runnable (fail open)")
        if r.returncode == 0:
            return Decision("allow", "green", (r.stdout or "").strip())
        reason = (r.stdout or r.stderr or "").strip()
        if r.returncode == 1 and profile == "medium":
            return Decision("downgrade", "yellow", reason or "thermal gate requested a lighter tier")
        return Decision("block", "red" if r.returncode == 2 else "yellow",
                        reason or f"thermal gate blocked profile {profile!r}")

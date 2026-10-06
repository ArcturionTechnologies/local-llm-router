"""Budget tracker: what did routing save, and what is left in each envelope?

Tracks three kinds of spend, keyed by each tier's ``cost`` class:

* ``paid``  -- converted to USD with the tier's ``price_in_per_mtok`` /
  ``price_out_per_mtok`` and compared with ``budget.paid_monthly_usd``.
* ``free``  -- raw tokens per UTC day, compared with the tier's
  ``daily_token_limit`` (free API quotas reset daily).
* ``local`` -- tokens that never left the machine ("displaced" from a paid API).

State lives in ``<state_dir>/budget-state.json`` and rolls over monthly. The
displacement report reads the router's telemetry log.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from .config import BudgetConfig, Config
from .errors import BudgetExceeded
from .telemetry import Telemetry


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def progress_bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct * width / 100)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


class Budget:
    def __init__(self, config: Config, now: Callable[[], datetime] = _utcnow):
        self.cfg: BudgetConfig = config.budget
        self.config = config
        self.path = config.state_path / "budget-state.json"
        self._now = now

    # -- state --------------------------------------------------------------
    def _blank(self) -> dict:
        return {"month": self._now().strftime("%Y-%m"), "tiers": {}}

    def load(self) -> dict:
        try:
            state = json.loads(self.path.read_text())
        except (OSError, ValueError):
            return self._blank()
        month = self._now().strftime("%Y-%m")
        if state.get("month") != month:
            return {"month": month, "previous_month": state.get("month"),
                    "previous_tiers": state.get("tiers"), "tiers": {}}
        return state

    def save(self, state: dict) -> None:
        state["updated_at"] = self._now().isoformat(timespec="seconds")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(state, indent=2))

    # -- recording ----------------------------------------------------------
    def record(self, tier_name: str, tokens_in: int, tokens_out: int) -> dict:
        tier = self.config.tier(tier_name)
        state = self.load()
        row = state["tiers"].setdefault(tier_name, {
            "calls": 0, "tokens_in": 0, "tokens_out": 0, "usd": 0.0,
            "day": "", "day_tokens": 0})
        today = self._now().strftime("%Y-%m-%d")
        if row.get("day") != today:
            row["day"], row["day_tokens"] = today, 0
        row["calls"] += 1
        row["tokens_in"] += tokens_in
        row["tokens_out"] += tokens_out
        row["day_tokens"] += tokens_in + tokens_out
        if tier.cost_class == "paid":
            row["usd"] += (tokens_in * tier.price_in_per_mtok
                           + tokens_out * tier.price_out_per_mtok) / 1_000_000.0
        self.save(state)
        return state

    # -- queries ------------------------------------------------------------
    def paid_used_usd(self, state: Optional[dict] = None) -> float:
        state = state or self.load()
        return sum(r.get("usd", 0.0) for n, r in state["tiers"].items()
                   if n in self.config.tiers and self.config.tiers[n].cost_class == "paid")

    def check(self, tier_name: str) -> tuple:
        """``("ok"|"warn"|"block", reason)`` for using this tier right now."""
        tier = self.config.tier(tier_name)
        state = self.load()
        if tier.cost_class == "paid" and self.cfg.paid_monthly_usd > 0:
            pct = self.paid_used_usd(state) / self.cfg.paid_monthly_usd * 100
            if pct >= self.cfg.block_pct:
                return "block", f"paid envelope {pct:.0f}% spent (${self.paid_used_usd(state):.2f} of ${self.cfg.paid_monthly_usd:.2f})"
            if pct >= self.cfg.warn_pct:
                return "warn", f"paid envelope {pct:.0f}% spent"
        if tier.cost_class == "free" and tier.daily_token_limit > 0:
            row = state["tiers"].get(tier_name, {})
            used = row.get("day_tokens", 0) if row.get("day") == self._now().strftime("%Y-%m-%d") else 0
            pct = used / tier.daily_token_limit * 100
            if pct >= 100:
                return "block", f"{tier_name} daily free quota used ({used:,} tokens)"
            if pct >= self.cfg.warn_pct:
                return "warn", f"{tier_name} daily free quota {pct:.0f}% used"
        return "ok", ""

    def enforce(self, tier_name: str) -> None:
        """Raise :class:`BudgetExceeded` when ``budget.enforce`` is on and the tier is blocked."""
        if not self.cfg.enforce:
            return
        verdict, reason = self.check(tier_name)
        if verdict == "block":
            raise BudgetExceeded(reason)

    def status(self, echo: Callable[[str], None] = print) -> dict:
        state = self.load()
        out = {"month": state["month"], "tiers": {}, "paid_used_usd": self.paid_used_usd(state),
               "paid_ceiling_usd": self.cfg.paid_monthly_usd}
        echo(f"LLM budget -- {state['month']}   (updated: {state.get('updated_at', 'never')})")
        paid = self.paid_used_usd(state)
        pct = paid / self.cfg.paid_monthly_usd * 100 if self.cfg.paid_monthly_usd else 0.0
        echo(f"  paid   ${paid:9.2f} / ${self.cfg.paid_monthly_usd:.2f}  {progress_bar(pct)} {pct:5.1f}%")
        for name, tier in self.config.tiers.items():
            row = state["tiers"].get(name)
            if not row:
                continue
            tokens = row["tokens_in"] + row["tokens_out"]
            extra = f"${row['usd']:.2f}" if tier.cost_class == "paid" else f"{tokens:,} tok"
            if tier.cost_class == "free" and tier.daily_token_limit:
                today = row["day_tokens"] if row.get("day") == self._now().strftime("%Y-%m-%d") else 0
                extra += f"  today {today:,}/{tier.daily_token_limit:,}"
            echo(f"  {tier.cost_class:5s}  {name:10s} {row['calls']:6,} calls  {extra}")
            out["tiers"][name] = row
        return out

    # -- displacement -------------------------------------------------------
    def displacement_report(self, telemetry: Optional[Telemetry] = None,
                            echo: Callable[[str], None] = print) -> dict:
        """How much work stayed local? Aggregates the telemetry log."""
        telemetry = telemetry or Telemetry(self.config.telemetry_path)
        calls = {"local": 0, "free": 0, "paid": 0, "unknown": 0}
        tokens = {"local": 0, "free": 0, "paid": 0, "unknown": 0}
        by_caller: dict = defaultdict(lambda: {"local": 0, "free": 0, "paid": 0, "unknown": 0})
        for e in telemetry.read():
            if e.get("ok") is False or "tier" not in e:
                continue
            bucket = e.get("cost") if e.get("cost") in calls else "unknown"
            calls[bucket] += 1
            tokens[bucket] += int(e.get("tokens_in") or 0) + int(e.get("tokens_out") or 0)
            by_caller[e.get("caller") or "unknown"][bucket] += 1
        total = sum(calls.values())
        paid_comparable = calls["local"] + calls["paid"]
        local_share = calls["local"] / total if total else 0.0
        displacement = calls["local"] / paid_comparable if paid_comparable else 0.0
        echo("Displacement report")
        echo(f"  local share:        {calls['local']:,} / {total:,} successful calls = {local_share * 100:.1f}%")
        echo(f"  paid displacement:  {calls['local']:,} local / {paid_comparable:,} paid-comparable = {displacement * 100:.1f}%")
        echo(f"  mix:                {calls['local']:,} local | {calls['free']:,} free cloud | {calls['paid']:,} paid cloud")
        echo(f"  tokens:             {tokens['local']:,} local | {tokens['free']:,} free | {tokens['paid']:,} paid")
        for caller, stats in sorted(by_caller.items(), key=lambda kv: -kv[1]["local"]):
            n = sum(stats.values())
            echo(f"    {caller:20s} {stats['local']:4d} local / {n:4d} total = {stats['local'] / n * 100:5.1f}%")
        return {"local_share": local_share, "paid_displacement": displacement,
                "calls": calls, "tokens": tokens, "by_caller": dict(by_caller)}

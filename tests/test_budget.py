"""Budget tracker: pricing, daily free quotas, month rollover, displacement report."""

import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from local_llm_router.budget import Budget, progress_bar
from local_llm_router.config import Config
from local_llm_router.errors import BudgetExceeded
from local_llm_router.telemetry import Telemetry


class Clock:
    def __init__(self, iso="2026-03-10T12:00:00+00:00"):
        self.t = datetime.fromisoformat(iso)

    def __call__(self):
        return self.t


class BudgetTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_budget_")
        self.cfg = Config(home=Path(self._tmp.name))
        self.clock = Clock()
        self.b = Budget(self.cfg, now=self.clock)

    def tearDown(self):
        self._tmp.cleanup()

    def test_paid_usage_is_converted_to_usd(self):
        self.b.record("sonnet", 1_000_000, 100_000)           # $3 in + $1.50 out
        self.assertAlmostEqual(self.b.paid_used_usd(), 4.5)

    def test_free_and_local_tiers_cost_nothing_but_are_counted(self):
        self.b.record("groq", 500, 500)
        self.b.record("small", 100, 100)
        state = self.b.load()
        self.assertEqual(state["tiers"]["groq"]["calls"], 1)
        self.assertEqual(state["tiers"]["small"]["tokens_in"], 100)
        self.assertEqual(self.b.paid_used_usd(), 0.0)

    def test_paid_envelope_thresholds(self):
        self.cfg.budget.paid_monthly_usd = 10.0
        self.assertEqual(self.b.check("sonnet")[0], "ok")
        self.b.record("sonnet", 2_000_000, 0)                 # $6 -> 60%
        self.assertEqual(self.b.check("sonnet")[0], "ok")
        self.b.record("sonnet", 1_000_000, 0)                 # $9 -> 90%
        self.assertEqual(self.b.check("sonnet")[0], "warn")
        self.b.record("sonnet", 1_000_000, 0)                 # $12 -> 120%
        verdict, reason = self.b.check("opus")
        self.assertEqual(verdict, "block")
        self.assertIn("paid envelope", reason)
        self.assertEqual(self.b.check("groq")[0], "ok")       # free tiers are not paid-envelope gated

    def test_enforce_only_raises_when_switched_on(self):
        self.cfg.budget.paid_monthly_usd = 1.0
        self.b.record("sonnet", 1_000_000, 0)
        self.b.enforce("sonnet")                              # enforce=False: no-op
        self.cfg.budget.enforce = True
        with self.assertRaises(BudgetExceeded):
            self.b.enforce("sonnet")
        self.b.enforce("small")                               # local tiers never blocked

    def test_free_tier_daily_quota_resets_each_day(self):
        self.cfg.tier("groq").daily_token_limit = 1000
        self.b.record("groq", 600, 400)
        self.assertEqual(self.b.check("groq")[0], "block")
        self.clock.t = datetime(2026, 3, 11, 0, 5, tzinfo=timezone.utc)
        self.assertEqual(self.b.check("groq")[0], "ok")
        self.b.record("groq", 100, 0)
        self.assertEqual(self.b.load()["tiers"]["groq"]["day_tokens"], 100)

    def test_month_rollover_archives_and_resets(self):
        self.b.record("sonnet", 1_000_000, 0)
        self.clock.t = datetime(2026, 4, 1, 0, 1, tzinfo=timezone.utc)
        state = self.b.load()
        self.assertEqual((state["month"], state["previous_month"]), ("2026-04", "2026-03"))
        self.assertEqual(state["tiers"], {})
        self.assertEqual(self.b.paid_used_usd(), 0.0)

    def test_corrupt_state_file_is_replaced_not_fatal(self):
        self.b.path.parent.mkdir(parents=True, exist_ok=True)
        self.b.path.write_text("{not json")
        self.assertEqual(self.b.load()["tiers"], {})

    def test_status_prints_a_summary(self):
        self.b.record("sonnet", 1000, 1000)
        lines = []
        out = self.b.status(echo=lines.append)
        self.assertIn("sonnet", out["tiers"])
        self.assertTrue(any("sonnet" in ln for ln in lines))

    def test_displacement_report_counts_local_share_by_caller(self):
        tel = Telemetry(Path(self._tmp.name) / "t.jsonl", env={})
        for cost, ok, caller in [("local", True, "job-a"), ("local", True, "job-a"),
                                 ("paid", True, "job-a"), ("free", True, "job-b"),
                                 ("local", False, "job-b")]:           # failed call must not count
            tel.log(tier="x", cost=cost, ok=ok, caller=caller, tokens_in=10, tokens_out=5)
        report = self.b.displacement_report(tel, echo=lambda s: None)
        self.assertEqual(report["calls"], {"local": 2, "free": 1, "paid": 1, "unknown": 0})
        self.assertAlmostEqual(report["local_share"], 0.5)
        self.assertAlmostEqual(report["paid_displacement"], 2 / 3)
        self.assertEqual(report["tokens"]["local"], 30)
        self.assertEqual(report["by_caller"]["job-a"]["local"], 2)

    def test_empty_report_is_zeroes_not_a_crash(self):
        report = self.b.displacement_report(Telemetry(Path(self._tmp.name) / "none.jsonl"), echo=lambda s: None)
        self.assertEqual(report["local_share"], 0.0)

    def test_progress_bar_clamps(self):
        self.assertEqual(progress_bar(-5, 4), "[----]")
        self.assertEqual(progress_bar(500, 4), "[####]")


if __name__ == "__main__":
    unittest.main(verbosity=2)

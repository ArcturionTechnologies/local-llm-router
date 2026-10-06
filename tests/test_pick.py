"""pick_tier(): the decision table that keeps bulk loops and low-stakes work off
paid tiers and on the local/free pool, while irreversible work gets the strongest tier."""

import unittest

from local_llm_router.config import Config
from local_llm_router.router import pick_tier

CFG = Config()


def tier(**kw) -> str:
    return pick_tier(CFG, **kw)[0]


class HardRules(unittest.TestCase):
    def test_irreversible_forces_strongest(self):
        self.assertEqual(tier(task_class="classify", stakes="irreversible"), "opus")
        # Even a trivial bulk classify cannot escape it.
        self.assertEqual(tier(task_class="classify", complexity="trivial",
                              stakes="irreversible", bulk_count=500), "opus")

    def test_long_context_forces_long_context_tier(self):
        self.assertEqual(tier(task_class="summarize", context_tokens=200_000), "gemini")

    def test_bulk_loop_never_uses_a_paid_tier(self):
        self.assertEqual(tier(task_class="classify", complexity="simple", bulk_count=100), "groq")
        self.assertEqual(tier(task_class="reason", complexity="complex", bulk_count=100), "cerebras")
        for t in ("groq", "cerebras"):
            self.assertNotEqual(CFG.tier(t).cost_class, "paid")

    def test_bulk_boundary_is_strictly_greater_than_threshold(self):
        # REGRESSION GUARD on the exact threshold: 20 is NOT bulk, 21 is.
        self.assertEqual(tier(task_class="classify", complexity="simple", bulk_count=20), "fast")
        self.assertEqual(tier(task_class="classify", complexity="simple", bulk_count=21), "groq")

    def test_context_boundary_is_strictly_greater_than_threshold(self):
        self.assertNotEqual(tier(task_class="summarize", context_tokens=100_000), "gemini")
        self.assertEqual(tier(task_class="summarize", context_tokens=100_001), "gemini")


class LocalFirst(unittest.TestCase):
    def test_vision_goes_to_the_vision_tier(self):
        for tc in ("vision", "ocr", "screenshot"):
            with self.subTest(tc=tc):
                self.assertEqual(tier(task_class=tc), "vision")

    def test_snap_classify_goes_to_the_fast_tier(self):
        self.assertEqual(tier(task_class="classify", complexity="simple"), "fast")
        self.assertEqual(tier(task_class="route", complexity="trivial"), "fast")

    def test_trivial_extract_goes_to_the_small_tier(self):
        self.assertEqual(tier(task_class="extract", complexity="trivial"), "small")

    def test_low_stakes_default_stays_local(self):
        self.assertEqual(tier(task_class="other", stakes="low"), "fast")
        self.assertEqual(tier(task_class="banana", stakes="low"), "fast")   # garbage in, safe out

    def test_default_medium_stakes_goes_to_strong_tier(self):
        self.assertEqual(tier(task_class="other", stakes="medium"), "sonnet")

    def test_code_gen_by_stakes(self):
        self.assertEqual(tier(task_class="code-gen", complexity="simple", stakes="low"), "coder")
        self.assertEqual(tier(task_class="code-gen", stakes="high"), "sonnet")
        self.assertEqual(tier(task_class="code-gen", stakes="irreversible"), "opus")

    def test_reasoning_by_stakes(self):
        self.assertEqual(tier(task_class="reason", complexity="moderate", stakes="low"), "medium")
        self.assertEqual(tier(task_class="reason", complexity="complex", stakes="high"), "sonnet")
        self.assertEqual(tier(task_class="reason", complexity="simple"), "small")

    def test_review_by_stakes(self):
        self.assertEqual(tier(task_class="review", stakes="high"), "opus")
        self.assertEqual(tier(task_class="review", stakes="low"), "medium")

    def test_complex_fallbacks(self):
        self.assertEqual(tier(task_class="other", complexity="complex", stakes="low"), "medium")
        self.assertEqual(tier(task_class="other", complexity="complex", stakes="high"), "opus")

    def test_returns_a_justification_string(self):
        name, why = pick_tier(CFG, task_class="classify", complexity="simple")
        self.assertEqual(name, "fast")
        self.assertTrue(why.strip())

    def test_every_pick_names_a_configured_tier(self):
        for tc in ("classify", "extract", "summarize", "route", "reason", "code-gen",
                   "edit", "analyze", "review", "vision", "other"):
            for cx in ("trivial", "simple", "moderate", "complex"):
                for sk in ("low", "medium", "high", "irreversible"):
                    self.assertIn(tier(task_class=tc, complexity=cx, stakes=sk), CFG.tiers)


class Remapping(unittest.TestCase):
    def test_roles_can_be_remapped(self):
        cfg = Config()
        cfg.roles["fast"] = "small"
        self.assertEqual(pick_tier(cfg, task_class="classify")[0], "small")

    def test_thresholds_can_be_changed(self):
        cfg = Config()
        cfg.routing.bulk_threshold = 5
        self.assertEqual(pick_tier(cfg, task_class="classify", bulk_count=6)[0], "groq")


if __name__ == "__main__":
    unittest.main(verbosity=2)

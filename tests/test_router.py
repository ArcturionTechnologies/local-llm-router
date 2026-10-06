"""Router integration tests: tier chains, thermal back-off, lock contention,
escalation, fallback modes, telemetry. Every backend is the mock HTTP server."""

import tempfile
import unittest
from pathlib import Path

import local_llm_router as llr
from helpers import FakeHost, MockLLMServer, QuietGate, TEST_ENV, make_config, make_router
from local_llm_router.errors import (AllTiersFailed, BudgetExceeded, RateLimited, ThermalBlocked,
                                     TierBusy, TierError)
from local_llm_router.model_lock import ModelLock
from local_llm_router.router import Router

BUSY = {"error": "local-pool-busy", "holder": "other-server", "holder_pid": 4242}


class RouterTestCase(unittest.TestCase):
    def setUp(self):
        self.srv = MockLLMServer("answer").start()
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_test_")
        self.tmp = Path(self._tmp.name)
        self.host = FakeHost()
        self.router = make_router(self.srv, self.tmp, self.host)

    def tearDown(self):
        self.srv.stop()
        self._tmp.cleanup()

    def models_hit(self) -> list:
        return [r["body"]["model"] for r in self.srv.posts()]


class HappyPath(RouterTestCase):
    def test_complete_returns_a_completion_with_metadata(self):
        c = self.router.complete("hello", tier="small")
        self.assertEqual((c.text, c.tier, str(c)), ("answer", "small", "answer"))
        self.assertEqual(c.model, self.router.config.tier("small").model)
        self.assertEqual((c.tokens_in, c.tokens_out), (11, 7))
        self.assertFalse(c.escalated)
        self.assertEqual([(a.tier, a.ok) for a in c.attempts], [("small", True)])

    def test_default_tier_comes_from_pick_tier(self):
        self.assertEqual(self.router.complete("x").tier, "fast")                         # low-stakes default
        self.assertEqual(self.router.complete("x", task_class="code-gen").tier, "coder")

    def test_chat_returns_text_only(self):
        self.assertEqual(self.router.chat("hello", tier="small"), "answer")

    def test_system_prompt_and_message_list(self):
        self.router.complete("hello", tier="small", system="be terse")
        self.assertEqual(self.srv.posts()[-1]["body"]["messages"][0],
                         {"role": "system", "content": "be terse"})
        self.router.complete([{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"},
                              {"role": "user", "content": "c"}], tier="small")
        self.assertEqual(len(self.srv.posts()[-1]["body"]["messages"]), 3)

    def test_unknown_tier_is_a_config_error(self):
        from local_llm_router.errors import ConfigError
        with self.assertRaises(ConfigError):
            self.router.complete("x", tier="nope")

    def test_health_reports_every_enabled_tier(self):
        h = self.router.health()
        self.assertTrue(all(h.values()))
        self.assertIn("small", h)
        self.assertIn("sonnet", h)

    def test_module_level_api_uses_the_default_router(self):
        llr.set_router(self.router)
        try:
            self.assertEqual(llr.chat("hi", tier="small"), "answer")
            self.assertEqual(llr.complete("hi", tier="small").tier, "small")
        finally:
            llr.set_router(None)


class ThermalBackoff(RouterTestCase):
    def test_red_blocks_local_tiers_and_the_call_falls_through_to_cloud(self):
        self.host.hot()
        c = self.router.complete("hi", tier="fast")
        self.assertEqual(c.tier, "groq")
        self.assertTrue(c.escalated)
        self.assertEqual([(a.tier, a.error) for a in c.attempts],
                         [("fast", "ThermalBlocked"), ("small", "ThermalBlocked"), ("groq", "")])
        self.assertEqual(self.models_hit(), [self.router.config.tier("groq").model])  # local servers never touched

    def test_yellow_downgrades_medium_to_small(self):
        self.host.busy()
        c = self.router.complete("hi", tier="medium")
        self.assertEqual(c.tier, "small")
        self.assertEqual(c.attempts[0].error, "ThermalDowngrade")
        self.assertEqual(len(self.srv.posts()), 1)

    def test_yellow_blocks_the_heavy_tier_and_steps_down_locally_first(self):
        self.host.busy()
        c = self.router.complete("hi", tier="heavy")
        self.assertEqual(c.tier, "small")         # heavy blocked -> medium downgraded -> small
        self.assertEqual([a.error for a in c.attempts], ["ThermalBlocked", "ThermalDowngrade", ""])

    def test_green_runs_the_requested_tier(self):
        self.assertEqual(self.router.complete("hi", tier="heavy").tier, "heavy")

    def test_ram_pressure_demotes_heavy_even_when_the_host_is_green(self):
        self.router.thermal.demote = True
        c = self.router.complete("hi", tier="heavy")
        self.assertEqual((c.tier, c.attempts[0].error), ("medium", "ThermalDowngrade"))

    def test_bypass_env_ignores_a_hot_host(self):
        env = {**TEST_ENV, "LLM_ROUTER_BYPASS_THERMAL": "1"}
        r = Router(self.router.config, thermal=QuietGate(self.router.config.thermal,
                   prober=self.host.hot(), env=env), env=env)
        self.assertEqual(r.complete("hi", tier="heavy").tier, "heavy")


class Failover(RouterTestCase):
    def test_busy_503_falls_through_to_the_next_tier(self):
        self.srv.respond(503, BUSY)
        c = self.router.complete("hi", tier="small")
        self.assertEqual(c.tier, "groq")
        self.assertEqual(c.attempts[0].error, "TierBusy")

    def test_transport_failure_falls_through(self):
        self.router.config.tier("small").base_url = "http://127.0.0.1:1/v1"
        c = self.router.complete("hi", tier="small", timeout=0.5)
        self.assertEqual((c.tier, c.attempts[0].error), ("groq", "TierError"))

    def test_fallback_raise_reraises_the_original_error_after_one_attempt(self):
        self.srv.respond(503, BUSY)
        with self.assertRaises(TierBusy):
            self.router.complete("hi", tier="small", fallback="raise")
        self.assertEqual(len(self.srv.posts()), 1)

    def test_fallback_skip_returns_an_empty_completion(self):
        self.host.hot()
        r = Router(self.router.config, thermal=self.router.thermal, env={})   # no cloud keys
        c = r.complete("hi", tier="fast", fallback="skip")
        self.assertEqual((c.text, c.tier), ("", None))
        self.assertEqual(len(c.attempts), 5)          # fast, small, groq, cerebras, gemini all refused

    def test_unknown_fallback_mode_fails_loudly(self):
        with self.assertRaises(ValueError):
            self.router.complete("hi", tier="small", fallback="groq ")           # trailing-space typo

    def test_all_tiers_failing_raises_with_the_attempt_trail(self):
        self.host.hot()
        r = Router(self.router.config, thermal=self.router.thermal, env={})
        with self.assertRaises(AllTiersFailed) as ctx:
            r.complete("hi", tier="small")
        self.assertEqual([a.tier for a in ctx.exception.attempts], ["small", "groq", "cerebras", "gemini"])
        self.assertIn("MissingCredentials", str(ctx.exception))

    def test_missing_cloud_key_is_skipped_without_a_request(self):
        r = Router(self.router.config, thermal=self.router.thermal, env={})
        self.host.hot()
        with self.assertRaises(AllTiersFailed):
            r.complete("hi", tier="groq")
        self.assertEqual(self.srv.posts(), [])

    def test_rate_limited_cloud_tier_is_remembered_and_skipped(self):
        self.srv.respond(429, {"error": "slow down"})
        c = self.router.complete("hi", tier="groq")
        self.assertEqual((c.tier, c.attempts[0].error), ("cerebras", "RateLimited"))
        hits = len(self.srv.posts())
        c2 = self.router.complete("hi", tier="groq")                              # cool-down: groq not contacted
        self.assertEqual((c2.tier, c2.attempts[0].error), ("cerebras", "RateLimited"))
        self.assertEqual(len(self.srv.posts()), hits + 1)

    def test_disabled_tiers_are_skipped(self):
        self.router.config.tier("small").enabled = False
        self.assertEqual([t.name for t in self.router.chain("fast")][:2], ["fast", "groq"])

    def test_chain_is_depth_first_and_deduplicated(self):
        names = [t.name for t in self.router.chain("heavy")]
        self.assertEqual(names, ["heavy", "medium", "small", "groq", "cerebras", "gemini"])


class ModelLockIntegration(RouterTestCase):
    def setUp(self):
        super().setUp()
        self.lock = ModelLock(self.tmp / "locks")
        self.router.lock = self.lock

    def test_exclusive_tier_holds_the_lock_only_during_the_call(self):
        self.assertEqual(self.router.complete("hi", tier="heavy").tier, "heavy")
        self.assertEqual(self.lock.holder(), {})

    def test_exclusive_tier_yields_when_another_process_holds_the_lock(self):
        other = ModelLock(self.tmp / "locks")
        self.assertTrue(other.acquire("other-server"))
        try:
            c = self.router.complete("hi", tier="heavy")
            self.assertEqual((c.tier, c.attempts[0].error), ("medium", "TierBusy"))
            self.assertEqual(self.models_hit(), [self.router.config.tier("medium").model])
        finally:
            other.release("other-server")
        self.assertEqual(self.router.complete("hi", tier="heavy").tier, "heavy")   # free again

    def test_lock_is_released_when_the_request_fails(self):
        self.srv.respond(500, {"error": "boom"})
        c = self.router.complete("hi", tier="heavy")
        self.assertEqual(c.attempts[0].error, "TierError")
        self.assertEqual(self.lock.holder(), {})

    def test_non_exclusive_tiers_ignore_the_lock(self):
        other = ModelLock(self.tmp / "locks")
        other.acquire("other-server")
        try:
            self.assertEqual(self.router.complete("hi", tier="small").tier, "small")
        finally:
            other.release("other-server")


class Escalation(RouterTestCase):
    def test_rejected_answer_escalates_to_the_next_tier(self):
        self.srv.respond(200, {"choices": [{"message": {"content": "meh"}}]})
        self.srv.respond(200, {"choices": [{"message": {"content": "a proper answer"}}]})
        c = self.router.complete("hard question", tier="small", accept=lambda t: len(t) > 10)
        self.assertEqual((c.text, c.tier, c.escalated), ("a proper answer", "groq", True))
        self.assertEqual([(a.tier, a.error) for a in c.attempts], [("small", "rejected"), ("groq", "")])

    def test_accepted_answer_does_not_escalate(self):
        c = self.router.complete("q", tier="small", accept=lambda t: True)
        self.assertEqual((c.tier, len(self.srv.posts())), ("small", 1))

    def test_everything_rejected_raises(self):
        with self.assertRaises(AllTiersFailed):
            self.router.complete("q", tier="small", accept=lambda t: False)

    def test_tier_call_trivial_hard_and_critical_use_the_configured_starting_tiers(self):
        for cx, tier, path in (("trivial", "fast", "/v1/chat/completions"),
                               ("routine", "medium", "/v1/chat/completions"),
                               ("hard", "sonnet", "/v1/messages"),
                               ("critical", "opus", "/v1/messages")):
            with self.subTest(complexity=cx):
                c = self.router.tier_call("do the thing", complexity=cx)
                self.assertEqual(c.tier, tier)
                self.assertEqual(self.srv.posts()[-1]["path"], path)

    def test_tier_call_auto_classifies_with_the_local_fast_tier_first(self):
        self.srv.respond(200, {"choices": [{"message": {"content": "hard"}}]})
        c = self.router.tier_call("design a distributed lock service")
        self.assertEqual(c.tier, "sonnet")
        self.assertEqual(self.srv.posts()[0]["body"]["model"], self.router.config.tier("fast").model)

    def test_tier_call_auto_defaults_to_routine_when_classification_fails(self):
        self.srv.respond(503, BUSY)
        self.assertEqual(self.router.tier_call("something").tier, "medium")

    def test_tier_call_auto_ignores_junk_classification(self):
        self.srv.respond(200, {"choices": [{"message": {"content": "banana"}}]})
        self.assertEqual(self.router.tier_call("something").tier, "medium")

    def test_complexity_mapping_is_configurable(self):
        self.router.config.complexity_tiers["routine"] = "sonnet"
        self.assertEqual(self.router.tier_call("x", complexity="routine").tier, "sonnet")

    def test_should_escalate(self):
        self.srv.reply_text = "ESCALATE"
        self.assertTrue(self.router.should_escalate("plan a migration"))
        self.srv.reply_text = "LOCAL"
        self.assertFalse(self.router.should_escalate("is this spam?"))


class Helpers(RouterTestCase):
    def say(self, text):
        self.srv.reply_text = text

    def test_classify_exact_match(self):
        self.say("earnings")
        self.assertEqual(self.router.classify("x", ["earnings", "macro", "noise"]), "earnings")

    def test_classify_normalises_quotes_case_and_sentences(self):
        self.say('The label is "Macro".')
        self.assertEqual(self.router.classify("x", ["earnings", "macro", "noise"]), "macro")

    def test_classify_junk_falls_back_to_the_last_label(self):
        self.say("completely unrelated gibberish")
        self.assertEqual(self.router.classify("x", ["earnings", "macro", "noise"]), "noise")

    def test_classify_requires_labels(self):
        with self.assertRaises(ValueError):
            self.router.classify("x", [])

    def test_classify_prefers_exact_match_over_prefix_shadow(self):
        self.say("earnings")
        self.assertEqual(self.router.classify("x", ["earn", "earnings"]), "earnings")

    def test_classify_word_boundary_beats_substring(self):
        self.say("this is macroeconomics")
        self.assertEqual(self.router.classify("x", ["macro", "macroeconomics"]), "macroeconomics")

    def test_extract_json_plain_fenced_and_surrounded(self):
        cases = {'{"a": 1}': {"a": 1},
                 '```json\n{"a": 1}\n```': {"a": 1},
                 '```\n{"a": 1}\n```': {"a": 1},
                 'Sure! Here you go: {"a": 1} Hope that helps.': {"a": 1}}
        for raw, want in cases.items():
            with self.subTest(raw=raw):
                self.say(raw)
                self.assertEqual(self.router.extract_json("x", {"a": "number"}), want)

    def test_extract_json_preserves_a_trailing_backtick_inside_a_value(self):
        self.say('```json\n{"cmd": "ls`"}\n```')
        self.assertEqual(self.router.extract_json("x", {"cmd": "string"}), {"cmd": "ls`"})

    def test_extract_json_without_json_is_an_error(self):
        self.say("I cannot do that")
        with self.assertRaises(TierError):
            self.router.extract_json("x", {"a": "number"})

    def test_summarize_passes_the_word_budget(self):
        self.say("short")
        self.assertEqual(self.router.summarize("long text", max_words=12), "short")
        self.assertIn("12 words", self.srv.posts()[-1]["body"]["messages"][0]["content"])

    def test_route_picks_one_option(self):
        self.say("billing")
        self.assertEqual(self.router.route("refund please", ["billing", "tech"]), "billing")

    def test_vision_inlines_images_and_never_falls_back(self):
        img = self.tmp / "shot.png"
        img.write_bytes(b"\x89PNG-test")
        self.router.vision("what is this?", [str(img)])
        content = self.srv.posts()[-1]["body"]["messages"][-1]["content"]
        self.assertEqual(content[0], {"type": "text", "text": "what is this?"})
        self.assertTrue(content[1]["image_url"]["url"].startswith("data:image/png;base64,"))
        self.host.hot()
        with self.assertRaises(ThermalBlocked):
            self.router.vision("again", [str(img)])

    def test_vision_rejects_a_missing_image_before_any_request(self):
        with self.assertRaises(TierError):
            self.router.vision("x", ["/nonexistent/nope.png"])
        self.assertEqual(self.srv.posts(), [])


class TelemetryAndBudget(RouterTestCase):
    def test_every_attempt_is_logged_including_failures(self):
        self.srv.respond(503, BUSY)
        self.router.complete("hi", tier="small", task="unit")
        rows = self.router.telemetry.read()
        self.assertEqual([(r["tier"], r["ok"]) for r in rows], [("small", False), ("groq", True)])
        self.assertEqual(rows[0]["error"], "TierBusy")
        self.assertTrue(rows[1]["fallback"])
        self.assertEqual((rows[1]["cost"], rows[1]["task"], rows[1]["tokens_in"]), ("free", "unit", 11))

    def test_caller_is_taken_from_the_environment(self):
        r = make_router(self.srv, self.tmp / "x", self.host)
        r.telemetry._env = {"LLM_ROUTER_CALLER": "Nightly-Job"}
        r.complete("hi", tier="small")
        self.assertEqual(r.telemetry.read()[0]["caller"], "nightly-job")

    def test_telemetry_can_be_disabled(self):
        self.router.config.telemetry.enabled = False
        self.router.telemetry.enabled = False
        self.router.complete("hi", tier="small")
        self.assertEqual(self.router.telemetry.read(), [])

    def test_paid_usage_is_priced_into_the_budget(self):
        self.router.complete("hi", tier="sonnet")          # mock reports 5 in / 3 out tokens
        state = self.router.budget.load()
        row = state["tiers"]["sonnet"]
        self.assertEqual((row["calls"], row["tokens_in"], row["tokens_out"]), (1, 5, 3))
        self.assertAlmostEqual(row["usd"], (5 * 3.0 + 3 * 15.0) / 1e6)

    def test_budget_enforcement_refuses_a_spent_paid_envelope(self):
        cfg = self.router.config
        cfg.budget.enforce, cfg.budget.paid_monthly_usd = True, 0.00001
        self.router.complete("hi", tier="sonnet")                   # spends > the tiny envelope
        r = Router(cfg, thermal=self.router.thermal, env=dict(TEST_ENV))
        with self.assertRaises(AllTiersFailed) as ctx:
            r.complete("hi", tier="sonnet")                         # sonnet and its opus fallback are both paid
        self.assertEqual({a.error for a in ctx.exception.attempts}, {"BudgetExceeded"})
        self.assertEqual(r.complete("hi", tier="groq").tier, "groq")   # free tiers unaffected


if __name__ == "__main__":
    unittest.main(verbosity=2)

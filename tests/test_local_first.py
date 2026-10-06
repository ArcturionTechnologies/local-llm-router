"""Local-first policy gate: warn once, then block routine work sent to paid APIs."""

import io
import json
import tempfile
import unittest
from pathlib import Path

from local_llm_router.config import Config
from local_llm_router.local_first import LocalFirstGate


class GateTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_lf_")
        self.cfg = Config(home=Path(self._tmp.name))
        self.gate = LocalFirstGate(self.cfg, env={"CLAUDE_SESSION_ID": "s1"})

    def tearDown(self):
        self._tmp.cleanup()

    def test_local_and_free_providers_always_pass(self):
        for p in ("local-qwen3-8b", "groq", "cerebras", "small"):
            with self.subTest(provider=p):
                self.assertEqual(self.gate.gate(p, "classify"), ("allow", None))

    def test_exempt_task_classes_pass(self):
        for tc in ("irreversible", "money-transfer", "external-send", "prod-deploy",
                   "schema-change", "security-audit", "legal-correspondence"):
            with self.subTest(task_class=tc):
                self.assertEqual(self.gate.gate("sonnet", tc)[0], "allow")

    def test_high_stakes_and_complex_work_pass(self):
        self.assertEqual(self.gate.gate("opus", "review", "simple", "irreversible")[0], "allow")
        self.assertEqual(self.gate.gate("opus", "review", "simple", "high")[0], "allow")
        self.assertEqual(self.gate.gate("sonnet", "code-gen", "complex", "low")[0], "allow")

    def test_routine_paid_call_warns_once_then_blocks(self):
        first = self.gate.gate("claude-haiku", "summarize", "trivial", "low")
        self.assertEqual(first[0], "warn")
        self.assertIn("first offense", first[1])
        second = self.gate.gate("claude-haiku", "summarize", "trivial", "low")
        third = self.gate.gate("sonnet", "classify", "simple", "low")
        self.assertEqual((second[0], third[0]), ("block", "block"))
        self.assertIn("violation #3", third[1])

    def test_violations_are_counted_per_session(self):
        self.gate.gate("sonnet", "classify")
        other = LocalFirstGate(self.cfg, env={"CLAUDE_SESSION_ID": "s2"})
        self.assertEqual(other.gate("sonnet", "classify")[0], "warn")

    def test_session_key_falls_back_to_the_day(self):
        self.assertTrue(LocalFirstGate(self.cfg, env={}).session_key().startswith("daily-"))

    def test_configured_paid_tiers_count_as_paid(self):
        self.assertTrue(self.gate.is_paid("haiku"))
        self.assertFalse(self.gate.is_paid("groq"))

    def test_clear_resets_the_counter(self):
        self.gate.gate("sonnet", "classify")
        self.gate.gate("sonnet", "classify")
        self.gate.clear()
        self.assertEqual(self.gate.gate("sonnet", "classify")[0], "warn")
        self.assertIn("1 violations", self.gate.status())

    def hook(self, event):
        out = io.StringIO()
        self.gate.hook(stdin=io.StringIO(json.dumps(event) if not isinstance(event, str) else event), stdout=out)
        return json.loads(out.getvalue())

    def test_hook_ignores_ordinary_commands_and_other_tools(self):
        self.assertEqual(self.hook({"tool_name": "Bash", "input": {"command": "ls -la"}}), {"decision": "allow"})
        self.assertEqual(self.hook({"tool_name": "Read", "input": {"command": "claude -p x"}}), {"decision": "allow"})

    def test_hook_warns_then_denies_paid_cli_calls_from_shell_tools(self):
        event = {"tool_name": "Bash", "input": {"command": 'claude -p "label this"'}}
        first = self.hook(event)
        self.assertEqual(first["decision"], "allow")
        self.assertIn("warning", first["message"])
        denied = self.hook(event)
        self.assertEqual(denied["decision"], "deny")
        self.assertIn("local-first policy", denied["reason"])

    def test_hook_passes_through_malformed_input(self):
        self.assertEqual(self.hook("not json"), {"decision": "allow"})

    def test_hook_detects_sdk_calls(self):
        ev = {"tool_name": "Bash", "input": {"command": "python -c 'anthropic.messages.create(...)'"}}
        self.assertEqual(self.hook(ev)["decision"], "allow")      # first offence warns


if __name__ == "__main__":
    unittest.main(verbosity=2)

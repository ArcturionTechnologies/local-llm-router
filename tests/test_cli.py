"""CLI end-to-end: real argument parsing, real config loading, mock model server."""

import contextlib
import io
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from helpers import MockLLMServer
from local_llm_router.cli import main


class CliTestCase(unittest.TestCase):
    def setUp(self):
        self.srv = MockLLMServer("pong").start()
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_cli_")
        self.home = Path(self._tmp.name)
        env = {"LLM_ROUTER_HOME": str(self.home),
               "LLM_ROUTER_LOCAL_BASE_URL": self.srv.v1,
               "LLM_ROUTER_BYPASS_THERMAL": "1",              # never read the real machine
               "LLM_ROUTER_URL_GROQ": self.srv.v1, "LLM_ROUTER_URL_CEREBRAS": self.srv.v1,
               "LLM_ROUTER_URL_GEMINI": self.srv.v1,
               "LLM_ROUTER_URL_SONNET": self.srv.url, "LLM_ROUTER_URL_HAIKU": self.srv.url,
               "LLM_ROUTER_URL_OPUS": self.srv.url,
               "GROQ_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"}
        patcher = mock.patch.dict(os.environ, env, clear=False)
        patcher.start()
        self.addCleanup(patcher.stop)
        for k in ("LLM_ROUTER_CONFIG", "LLM_ROUTER_STATE_DIR", "LLM_ROUTER_CALLER"):
            os.environ.pop(k, None)

    def tearDown(self):
        self.srv.stop()
        self._tmp.cleanup()

    def run_cli(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def write_config(self, text):
        (self.home / "config.toml").write_text(text)


class ModelCommands(CliTestCase):
    def test_chat(self):
        code, out, _ = self.run_cli("chat", "ping", "--tier", "small")
        self.assertEqual((code, out.strip()), (0, "pong"))

    def test_chat_verbose_reports_the_tier_on_stderr(self):
        code, out, err = self.run_cli("chat", "ping", "--tier", "small", "-v")
        self.assertEqual(out.strip(), "pong")
        self.assertIn("tier=small", err)

    def test_classify_summarize_route_extract(self):
        self.srv.reply_text = "billing"
        self.assertEqual(self.run_cli("classify", "refund me", "billing,tech")[1].strip(), "billing")
        self.assertEqual(self.run_cli("route", "refund me", "billing,tech")[1].strip(), "billing")
        self.srv.reply_text = "a summary"
        self.assertEqual(self.run_cli("summarize", "long text", "10")[1].strip(), "a summary")
        self.srv.reply_text = '{"vendor": "Acme", "amount": 12}'
        code, out, _ = self.run_cli("extract", "Acme invoice 12", '{"vendor": "string", "amount": "number"}')
        self.assertEqual((code, json.loads(out)), (0, {"vendor": "Acme", "amount": 12}))

    def test_escalate_prints_yes_or_no(self):
        self.srv.reply_text = "ESCALATE"
        self.assertEqual(self.run_cli("escalate", "plan a migration")[1].strip(), "YES")
        self.srv.reply_text = "LOCAL"
        self.assertEqual(self.run_cli("escalate", "spam?")[1].strip(), "NO")

    def test_tier_call_with_explicit_complexity(self):
        code, out, err = self.run_cli("tier-call", "hard thing", "--complexity", "hard", "-v")
        self.assertEqual((code, out.strip()), (0, "pong"))
        self.assertIn("tier=sonnet", err)

    def test_total_failure_exits_2_with_a_message(self):
        os.environ.pop("LLM_ROUTER_LOCAL_BASE_URL")          # env would otherwise override the file
        self.write_config('[tiers.small]\nbase_url = "http://127.0.0.1:1/v1"\nfallbacks = []\n')
        code, out, err = self.run_cli("chat", "ping", "--tier", "small", "--fallback", "raise")
        self.assertEqual((code, out), (2, ""))
        self.assertIn("[error]", err)

    def test_bad_json_schema_is_a_bad_args_error(self):
        self.assertEqual(self.run_cli("extract", "x", "{not json")[0], 1)

    def test_health_exit_code_reflects_reachability(self):
        code, out, _ = self.run_cli("health")
        self.assertEqual(code, 0)
        self.assertIn("UP   small", out)
        self.write_config("".join(f'[tiers.{n}]\nbase_url = "http://127.0.0.1:1/v1"\n'
                                  for n in ("fast", "small", "medium", "coder", "heavy", "vision")))
        with mock.patch.dict(os.environ, {"LLM_ROUTER_LOCAL_BASE_URL": "http://127.0.0.1:1/v1"}):
            os.environ.pop("GROQ_API_KEY"); os.environ.pop("ANTHROPIC_API_KEY")
            code, out, _ = self.run_cli("health")
        self.assertEqual(code, 2)
        self.assertIn("DOWN small", out)


class OfflineCommands(CliTestCase):
    def test_pick(self):
        code, out, _ = self.run_cli("pick", "--task-class", "classify", "--bulk", "100")
        self.assertEqual((code, out.split("\t")[0]), (0, "groq"))

    def test_tiers_lists_chains(self):
        out = self.run_cli("tiers")[1]
        self.assertIn("chain: heavy -> medium -> groq", out)

    def test_config_init_show_path(self):
        code, out, _ = self.run_cli("config", "init")
        self.assertEqual(code, 0)
        self.assertTrue((self.home / "config.toml").is_file())
        self.assertEqual(self.run_cli("config", "init")[0], 1)               # refuses to overwrite
        self.assertEqual(self.run_cli("config", "init", "--force")[0], 0)
        self.assertEqual(self.run_cli("config", "path")[1].strip(), str(self.home / "config.toml"))
        shown = json.loads(self.run_cli("config", "show")[1])
        self.assertIn("fast", shown["tiers"])
        # The generated file is itself a valid config.
        os.environ.pop("LLM_ROUTER_LOCAL_BASE_URL")
        self.assertEqual(self.run_cli("tiers")[0], 0)

    def test_invalid_config_exits_1(self):
        self.write_config("[thermal]\nbogus = 1\n")
        code, _, err = self.run_cli("tiers")
        self.assertEqual(code, 1)
        self.assertIn("unknown key", err)

    def test_thermal_exit_codes_follow_the_gate_contract(self):
        self.assertEqual(self.run_cli("thermal", "--profile", "heavy")[0], 0)      # bypass env set
        del os.environ["LLM_ROUTER_BYPASS_THERMAL"]
        gate = self.home / "gate.sh"
        for exit_code, want in ((0, 0), (1, 1), (2, 2)):
            gate.write_text(f"#!/bin/bash\necho verdict-{exit_code}\nexit {exit_code}\n")
            gate.chmod(gate.stat().st_mode | stat.S_IEXEC)
            self.write_config(f'[thermal]\ncommand = "{gate}"\n')
            code, out, _ = self.run_cli("thermal", "--profile", "medium")
            self.assertEqual(code, want)
            self.assertIn(f"verdict-{exit_code}" if exit_code else "ALLOW", out)

    def test_budget_record_and_status(self):
        self.assertEqual(self.run_cli("budget", "record", "sonnet", "1000000", "0")[0], 0)
        out = self.run_cli("budget", "status")[1]
        self.assertIn("sonnet", out)
        self.assertIn("$     3.00", out)
        self.assertEqual(self.run_cli("budget", "displacement")[0], 0)
        self.assertEqual(self.run_cli("budget", "snapshot")[0], 0)

    def test_lock_status_and_force_release(self):
        self.assertIn("not held", self.run_cli("lock", "status")[1])
        self.assertEqual(self.run_cli("lock", "force-release")[0], 0)

    def test_guard_check_and_log_spawn(self):
        self.write_config('[rate_guard]\nprocess_pattern = ""\n')
        code, out, _ = self.run_cli("guard", "check")
        self.assertEqual((code, json.loads(out)["status"]), (0, "ok"))
        self.assertEqual(self.run_cli("guard", "log-spawn", "nightly", "t1")[0], 0)
        (self.home / "state" / "KILL_ACTIVE").write_text("")
        self.assertEqual(self.run_cli("guard", "check")[0], 2)

    def test_heartbeat_exit_codes(self):
        watched = self.home / "inbox.md"
        watched.write_text("x")
        sig = self.home / "heartbeat" / "signals"
        sig.mkdir(parents=True)
        (sig / "inbox.json").write_text(json.dumps({"files": [str(watched)]}))
        self.assertEqual(self.run_cli("heartbeat", "should-fire", "inbox")[0], 0)   # first tick fires
        self.assertEqual(self.run_cli("heartbeat", "should-fire", "inbox")[0], 1)   # nothing changed
        self.assertEqual(self.run_cli("heartbeat", "should-fire", "ghost")[0], 1)   # no manifest
        self.assertEqual(self.run_cli("heartbeat", "reset", "inbox")[0], 0)
        self.assertEqual(self.run_cli("heartbeat", "should-fire", "inbox")[0], 0)
        self.assertTrue(json.loads(self.run_cli("heartbeat", "status", "inbox")[1])["manifest_present"])
        self.assertEqual(self.run_cli("heartbeat", "should-fire")[0], 1)            # missing id

    def test_local_first_test_matrix_and_clear(self):
        out = self.run_cli("local-first", "test")[1]
        self.assertIn("allow", out)
        self.assertEqual(self.run_cli("local-first", "clear")[0], 0)
        self.assertIn("No local-first violations", self.run_cli("local-first", "status")[1])

    def test_no_command_prints_help(self):
        code, out, _ = self.run_cli()
        self.assertEqual(code, 0)
        self.assertIn("usage: llm-router", out)


if __name__ == "__main__":
    unittest.main(verbosity=2)

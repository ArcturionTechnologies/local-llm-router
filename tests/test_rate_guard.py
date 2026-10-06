"""Session rate guard: ceilings, cool-down, kill switch, log scanning."""

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from local_llm_router.config import Config
from local_llm_router.rate_guard import RateGuard


class Clock:
    def __init__(self):
        self.t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t


class GuardTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_guard_")
        self.tmp = Path(self._tmp.name)
        self.cfg = Config(home=self.tmp)
        self.clock = Clock()
        self.active = 0
        self.guard = RateGuard(self.cfg, now=self.clock, active_counter=lambda: self.active)

    def tearDown(self):
        self._tmp.cleanup()

    def spawn(self, n=1, minutes_ago=0, event="spawn_fired"):
        self.guard.manifest.parent.mkdir(parents=True, exist_ok=True)
        ts = (self.clock.t - timedelta(minutes=minutes_ago)).isoformat(timespec="seconds")
        with self.guard.manifest.open("a") as f:
            for i in range(n):
                f.write(json.dumps({"ts": ts, "event": event, "caller": "t", "task_id": str(i)}) + "\n")

    def test_idle_is_ok(self):
        rec = self.guard.check()
        self.assertEqual((rec["status"], self.guard.exit_code(rec)), ("ok", 0))

    def test_concurrency_ceiling_blocks(self):
        self.active = self.cfg.rate_guard.max_concurrent
        rec = self.guard.check()
        self.assertEqual((rec["status"], self.guard.exit_code(rec)), ("block", 2))
        self.assertIn("concurrent", rec["reason"])

    def test_hourly_ceiling_walks_ok_throttle_block(self):
        self.cfg.rate_guard.max_1h, self.cfg.rate_guard.max_5h = 10, 100
        self.spawn(7)
        self.assertEqual(self.guard.check()["status"], "ok")
        self.spawn(1)                                        # 8/10 = 80% -> yellow
        rec = self.guard.check()
        self.assertEqual((rec["status"], self.guard.exit_code(rec)), ("throttle", 1))
        self.spawn(2)                                        # 10/10 -> red
        self.assertEqual(self.guard.check()["status"], "block")

    def test_old_spawns_age_out_of_the_hour_window_but_count_in_the_week(self):
        self.spawn(5, minutes_ago=90)
        self.assertEqual(self.guard.count_spawns(1), 0)
        self.assertEqual(self.guard.count_spawns(5), 5)
        self.assertEqual(self.guard.count_spawns(168), 5)

    def test_weekly_ceiling_blocks_even_when_the_hour_is_quiet(self):
        self.cfg.rate_guard.max_week = 10
        self.spawn(10, minutes_ago=60 * 24 * 3)
        rec = self.guard.check()
        self.assertEqual(rec["status"], "block")
        self.assertEqual(rec["spawns_1h"], 0)

    def test_only_spawn_events_count(self):
        self.spawn(5, event="spawn_denied")
        self.assertEqual(self.guard.count_spawns(1), 0)

    def test_log_spawn_round_trips(self):
        self.guard.log_spawn("nightly", "task-1")
        self.assertEqual(self.guard.count_spawns(1), 1)

    def test_registry_events_are_counted_once_configured(self):
        reg = self.tmp / "registry.jsonl"
        reg.write_text(json.dumps({"ts": self.clock.t.isoformat(timespec="seconds"),
                                   "event": "session_start"}) + "\n")
        self.assertEqual(self.guard.count_spawns(1), 0)
        self.cfg.rate_guard.registry_path = str(reg)
        self.assertEqual(self.guard.count_spawns(1), 1)

    def test_cooldown_throttles_then_expires(self):
        self.guard.set_cooldown("test")
        rec = self.guard.check()
        self.assertEqual((rec["status"], rec["cooldown"]), ("throttle", True))
        self.assertGreater(rec["cooldown_remaining_min"], 0)
        self.clock.t += timedelta(minutes=self.cfg.rate_guard.cooldown_min + 1)
        self.assertEqual(self.guard.check()["status"], "ok")

    def test_rate_limit_text_in_recent_logs_sets_a_cooldown(self):
        logs = self.tmp / "spawn-logs"
        logs.mkdir()
        (logs / "run1.log").write_text("working...\nAPI error: 429 Too Many Requests\n")
        self.cfg.rate_guard.spawn_log_dir = str(logs)
        rec = self.guard.check()
        self.assertEqual(rec["status"], "throttle")
        self.assertTrue(self.guard.in_cooldown()[0])

    def test_clean_logs_do_not_trigger(self):
        logs = self.tmp / "spawn-logs"
        logs.mkdir()
        (logs / "run1.log").write_text("all good\n")
        self.cfg.rate_guard.spawn_log_dir = str(logs)
        self.assertEqual(self.guard.check()["status"], "ok")

    def test_kill_switch_beats_everything(self):
        (self.cfg.state_path).mkdir(parents=True)
        (self.cfg.state_path / "KILL_ACTIVE").write_text("")
        rec = self.guard.check()
        self.assertEqual((rec["status"], self.guard.exit_code(rec)), ("block", 2))
        self.assertIn("KILL SWITCH", rec["reason"])

    def test_extra_kill_flags_are_honoured(self):
        flag = self.tmp / "stop-everything"
        flag.write_text("")
        self.cfg.rate_guard.kill_flags = [str(flag)]
        guard = RateGuard(self.cfg, now=self.clock, active_counter=lambda: 0)
        self.assertEqual(guard.check()["status"], "block")

    def test_state_file_is_written_for_dashboards(self):
        self.guard.check()
        self.assertEqual(json.loads(self.guard.state_file.read_text())["status"], "ok")

    def test_active_session_count_uses_pgrep_with_the_configured_pattern(self):
        calls = []

        class R:
            stdout = "101\n102\n"

        def run(cmd, **kw):
            calls.append(cmd)
            return R()
        guard = RateGuard(self.cfg, run=run)
        self.assertEqual(guard.count_active(), 2)
        self.assertEqual(calls[0], ["pgrep", "-f", "claude -p"])
        self.cfg.rate_guard.process_pattern = ""
        self.assertEqual(guard.count_active(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

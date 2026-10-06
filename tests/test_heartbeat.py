"""Heartbeat gate: fire only when signals change, sanity floor, fail-closed, thermal back-off."""

import json
import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

from helpers import FakeHost
from local_llm_router.config import Config
from local_llm_router.heartbeat import HeartbeatGate, compute_fingerprint
from local_llm_router.thermal import ThermalGate


class Clock:
    def __init__(self):
        self.t = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)

    def __call__(self):
        return self.t

    def advance(self, **kw):
        self.t += timedelta(**kw)


class HeartbeatTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_hb_")
        self.tmp = Path(self._tmp.name)
        self.cfg = Config(home=self.tmp)
        self.clock = Clock()
        self.host = FakeHost()
        self.gate = HeartbeatGate(self.cfg, now=self.clock,
                                  thermal=ThermalGate(self.cfg.thermal, prober=self.host, env={}))
        self.watched = self.tmp / "inbox.md"
        self.watched.write_text("one")

    def tearDown(self):
        self._tmp.cleanup()

    def manifest(self, hbid="inbox", **extra):
        self.gate.signals_dir.mkdir(parents=True, exist_ok=True)
        doc = {"files": [str(self.watched)], "sanity_fire_hours": 6, **extra}
        (self.gate.signals_dir / f"{hbid}.json").write_text(json.dumps(doc))

    def touch(self, text):
        self.watched.write_text(text)
        st = self.watched.stat()
        os.utime(self.watched, (st.st_atime, st.st_mtime + 5))   # mtime has 1 s resolution

    def test_no_manifest_fails_closed(self):
        self.assertFalse(self.gate.should_fire("ghost"))

    def test_first_tick_fires_then_quiet_ticks_skip(self):
        self.manifest()
        self.assertTrue(self.gate.should_fire("inbox"))
        self.clock.advance(minutes=5)
        self.assertFalse(self.gate.should_fire("inbox"))
        self.clock.advance(minutes=5)
        self.assertFalse(self.gate.should_fire("inbox"))
        self.assertEqual(self.gate.load_state("inbox")["skip_count_since_last_fire"], 2)

    def test_a_changed_file_fires_again(self):
        self.manifest()
        self.gate.should_fire("inbox")
        self.touch("two, longer")
        self.clock.advance(minutes=5)
        self.assertTrue(self.gate.should_fire("inbox"))
        self.assertEqual(self.gate.load_state("inbox")["last_fire_reason"], "fingerprint_changed")

    def test_sanity_floor_fires_even_when_nothing_changed(self):
        self.manifest(sanity_fire_hours=2)
        self.gate.should_fire("inbox")
        self.clock.advance(hours=1)
        self.assertFalse(self.gate.should_fire("inbox"))
        self.clock.advance(hours=1.5)
        self.assertTrue(self.gate.should_fire("inbox"))
        self.assertEqual(self.gate.load_state("inbox")["last_fire_reason"], "sanity_floor")

    def test_directory_and_command_signals(self):
        d = self.tmp / "approvals"
        d.mkdir()
        self.manifest(dirs=[str(d)], commands=[{"name": "n", "cmd": f"ls {d} | wc -l"}], files=[])
        self.assertTrue(self.gate.should_fire("inbox"))
        self.clock.advance(minutes=5)
        self.assertFalse(self.gate.should_fire("inbox"))
        (d / "new.txt").write_text("x")
        self.clock.advance(minutes=5)
        self.assertTrue(self.gate.should_fire("inbox"))

    def test_missing_files_are_a_stable_signal_not_an_error(self):
        fp1 = compute_fingerprint({"files": [str(self.tmp / "nope")]})
        self.assertEqual(fp1, compute_fingerprint({"files": [str(self.tmp / "nope")]}))
        (self.tmp / "nope").write_text("now it exists")
        self.assertNotEqual(fp1, compute_fingerprint({"files": [str(self.tmp / "nope")]}))

    def test_reset_forces_the_next_tick(self):
        self.manifest()
        self.gate.should_fire("inbox")
        self.clock.advance(minutes=1)
        self.assertFalse(self.gate.should_fire("inbox"))
        self.gate.reset("inbox")
        self.assertTrue(self.gate.should_fire("inbox"))

    def test_hot_host_backs_off_without_consuming_the_change(self):
        self.manifest(skip_when_hot=True)
        self.host.hot()
        self.assertFalse(self.gate.should_fire("inbox"))
        self.assertEqual(self.gate.load_state("inbox")["last_skip_reason"], "host_hot")
        self.host.set(cpu_load=0.5)                       # cooled down
        self.clock.advance(minutes=5)
        self.assertTrue(self.gate.should_fire("inbox"))   # the first fire was deferred, not lost

    def test_hot_host_is_ignored_unless_the_manifest_opts_in(self):
        self.manifest()
        self.host.hot()
        self.assertTrue(self.gate.should_fire("inbox"))

    def test_log_path_receives_decisions(self):
        log = self.tmp / "logs" / "hb.log"
        self.manifest(log_path=str(log))
        self.gate.should_fire("inbox")
        self.clock.advance(minutes=1)
        self.gate.should_fire("inbox")
        lines = log.read_text().splitlines()
        self.assertIn("firing: first_fire", lines[0])
        self.assertIn("skip #1", lines[1])

    def test_status_and_discover(self):
        self.manifest()
        self.gate.should_fire("inbox")
        st = self.gate.status("inbox")
        self.assertTrue(st["manifest_present"])
        self.assertEqual(st["current_fingerprint"], st["state"]["last_fingerprint"])
        tasks = self.tmp / "tasks"
        (tasks / "inbox").mkdir(parents=True)
        (tasks / "orphan").mkdir()
        self.cfg.heartbeat.tasks_dir = str(tasks)
        gate = HeartbeatGate(self.cfg, now=self.clock)
        self.assertEqual(gate.discover(), [{"hbid": "inbox", "has_manifest": True},
                                           {"hbid": "orphan", "has_manifest": False}])

    def test_corrupt_manifest_and_state_are_tolerated(self):
        self.gate.signals_dir.mkdir(parents=True)
        (self.gate.signals_dir / "bad.json").write_text("{nope")
        self.assertFalse(self.gate.should_fire("bad"))
        self.manifest()
        self.gate.state_dir.mkdir(parents=True)
        (self.gate.state_dir / "inbox.json").write_text("garbage")
        self.assertTrue(self.gate.should_fire("inbox"))


if __name__ == "__main__":
    unittest.main(verbosity=2)

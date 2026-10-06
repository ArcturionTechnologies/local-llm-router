"""Single-heavy-model lock: exclusion, metadata, stale holders, context manager."""

import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

from local_llm_router.errors import TierBusy
from local_llm_router.model_lock import ModelLock


class LockTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="llmrouter_lock_")
        self.dir = Path(self._tmp.name) / "locks"
        self.a, self.b = ModelLock(self.dir), ModelLock(self.dir)   # two "processes"

    def tearDown(self):
        self.a.release("model-a")
        self.b.release("model-b")
        self._tmp.cleanup()

    def test_acquire_excludes_a_second_holder(self):
        self.assertTrue(self.a.acquire("model-a"))
        self.assertFalse(self.b.acquire("model-b"))

    def test_holder_metadata_names_the_server_and_pid(self):
        self.a.acquire("model-a")
        h = self.b.holder()
        self.assertEqual((h["server"], h["pid"]), ("model-a", os.getpid()))
        self.assertTrue(self.b.is_held_by_other("model-b"))
        self.assertFalse(self.b.is_held_by_other("model-a"))

    def test_release_frees_the_lock_and_is_idempotent(self):
        self.a.acquire("model-a")
        self.a.release("model-a")
        self.a.release("model-a")
        self.assertEqual(self.a.holder(), {})
        self.assertTrue(self.b.acquire("model-b"))

    def test_reacquire_by_the_same_holder_is_true_by_another_name_false(self):
        self.assertTrue(self.a.acquire("model-a"))
        self.assertTrue(self.a.acquire("model-a"))
        self.assertFalse(self.a.acquire("model-b"))

    def test_dead_pid_metadata_is_treated_as_unheld(self):
        self.dir.mkdir(parents=True)
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()                                           # pid is now dead
        (self.dir / "local-model.meta.json").write_text(json.dumps({"server": "ghost", "pid": proc.pid}))
        self.assertEqual(self.a.holder(), {})
        self.assertTrue(self.a.acquire("model-a"))            # stale metadata does not block

    def test_lock_dies_with_its_process(self):
        code = ("import sys, time; from local_llm_router.model_lock import ModelLock; "
                f"l = ModelLock({str(self.dir)!r}); assert l.acquire('child'); print('held', flush=True); time.sleep(30)")
        child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True,
                                 cwd=Path(__file__).resolve().parent.parent)
        try:
            self.assertEqual(child.stdout.readline().strip(), "held")
            self.assertFalse(self.a.acquire("model-a"))       # a real second process holds it
            self.assertEqual(self.a.holder()["server"], "child")
            child.kill()
            child.wait()
            for _ in range(50):                               # kernel drops the flock on exit
                if self.a.acquire("model-a"):
                    break
                time.sleep(0.05)
            else:
                self.fail("lock not released after the holder died")
        finally:
            child.kill()

    def test_hold_context_manager_raises_tierbusy_with_holder_details(self):
        with self.a.hold("model-a"):
            with self.assertRaises(TierBusy) as ctx:
                with self.b.hold("model-b"):
                    self.fail("should not enter")
            self.assertEqual(ctx.exception.holder, "model-a")
        with self.b.hold("model-b"):                          # free after the first block exits
            self.assertEqual(self.a.holder()["server"], "model-b")

    def test_hold_releases_on_exception(self):
        with self.assertRaises(RuntimeError):
            with self.a.hold("model-a"):
                raise RuntimeError("boom")
        self.assertEqual(self.a.holder(), {})

    def test_force_release_clears_metadata(self):
        self.a.acquire("model-a")
        self.b.force_release()
        self.assertEqual(self.b.holder(), {})


if __name__ == "__main__":
    unittest.main(verbosity=2)

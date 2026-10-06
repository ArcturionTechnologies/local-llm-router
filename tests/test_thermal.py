"""Thermal gate: green/yellow/red x profile decision table, external-command
contract, RAM-pressure demotion. Host readings are faked; nothing probes the machine."""

import stat
import subprocess
import tempfile
import unittest
from pathlib import Path

from helpers import FakeHost
from local_llm_router.config import ThermalConfig
from local_llm_router.errors import ThermalBlocked, ThermalDowngrade
from local_llm_router.thermal import (Snapshot, ThermalGate, _free_pct_from_vm_stat, classify,
                                      decide, free_memory_gb, should_demote_heavy)

CFG = ThermalConfig()
ALL = ("light", "standard", "medium", "coder", "vision", "heavy")


def snap(**kw) -> Snapshot:
    return FakeHost(**kw).snapshot


class Classification(unittest.TestCase):
    def test_green(self):
        self.assertEqual(classify(snap(), CFG)[0], "green")

    def test_each_red_trigger(self):
        for kw in (dict(cpu_load=8.5), dict(mem_free_pct=7.0),
                   dict(on_battery=True, battery_pct=10), dict(speed_limit=70)):
            with self.subTest(kw=kw):
                self.assertEqual(classify(snap(**kw), CFG)[0], "red")

    def test_each_yellow_trigger(self):
        for kw in (dict(cpu_load=5.5), dict(mem_free_pct=12.0)):
            with self.subTest(kw=kw):
                self.assertEqual(classify(snap(**kw), CFG)[0], "yellow")

    def test_battery_only_matters_when_unplugged(self):
        self.assertEqual(classify(snap(on_battery=False, battery_pct=5), CFG)[0], "green")

    def test_unreadable_sensors_are_never_green(self):
        self.assertEqual(classify(snap(cpu_load=None), CFG)[0], "unknown")
        self.assertEqual(classify(snap(thermal_known=False), CFG)[0], "unknown")

    def test_red_outranks_unknown(self):
        self.assertEqual(classify(snap(cpu_load=9.9, thermal_known=False), CFG)[0], "red")


class DecisionTable(unittest.TestCase):
    def test_green_allows_every_profile(self):
        for p in ALL:
            with self.subTest(profile=p):
                self.assertEqual(decide(snap(), p, CFG).action, "allow")

    def test_red_blocks_every_gated_profile_but_never_softens_to_downgrade(self):
        # REGRESSION GUARD: red must not be turned into a "use a lighter model" hint.
        for p in ALL:
            with self.subTest(profile=p):
                d = decide(snap(cpu_load=9.9), p, CFG)
                self.assertEqual((d.action, d.level), ("block", "red"))

    def test_red_does_not_gate_ungated_tiers(self):
        self.assertEqual(decide(FakeHost().hot().snapshot, "none", CFG).action, "allow")

    def test_yellow_per_profile(self):
        host = FakeHost().busy().snapshot
        expect = {"light": "allow", "standard": "allow", "medium": "downgrade",
                  "coder": "block", "vision": "block", "heavy": "block"}
        for p, action in expect.items():
            with self.subTest(profile=p):
                self.assertEqual(decide(host, p, CFG).action, action)

    def test_unknown_sensors_per_profile(self):
        host = snap(thermal_known=False)
        expect = {"light": "allow", "standard": "block", "medium": "downgrade",
                  "coder": "block", "vision": "block", "heavy": "block"}
        for p, action in expect.items():
            with self.subTest(profile=p):
                self.assertEqual(decide(host, p, CFG).action, action)

    def test_exit_codes(self):
        self.assertEqual(decide(snap(), "light", CFG).exit_code, 0)
        self.assertEqual(decide(FakeHost().busy().snapshot, "medium", CFG).exit_code, 1)
        self.assertEqual(decide(FakeHost().hot().snapshot, "light", CFG).exit_code, 2)

    def test_thresholds_are_configurable(self):
        strict = ThermalConfig(cpu_moderate_ratio=0.05)
        self.assertEqual(decide(snap(cpu_load=1.0), "medium", strict).action, "downgrade")


class GateObject(unittest.TestCase):
    def test_enforce_raises_the_right_exception(self):
        host = FakeHost()
        gate = ThermalGate(CFG, prober=host, env={})
        gate.enforce("medium")                                   # green: no exception
        host.busy()
        with self.assertRaises(ThermalDowngrade):
            gate.enforce("medium")
        with self.assertRaises(ThermalBlocked):
            gate.enforce("heavy")
        host.hot()
        with self.assertRaises(ThermalBlocked):
            gate.enforce("medium")

    def test_downgrade_and_block_are_distinct_types(self):
        self.assertFalse(issubclass(ThermalDowngrade, ThermalBlocked))
        self.assertFalse(issubclass(ThermalBlocked, ThermalDowngrade))

    def test_disabled_gate_allows_everything(self):
        gate = ThermalGate(ThermalConfig(enabled=False), prober=FakeHost().hot(), env={})
        self.assertEqual(gate.check("heavy").action, "allow")

    def test_bypass_env_skips_the_gate(self):
        gate = ThermalGate(CFG, prober=FakeHost().hot(), env={"LLM_ROUTER_BYPASS_THERMAL": "1"})
        self.assertEqual(gate.check("heavy").action, "allow")
        self.assertFalse(gate.should_demote())


def _script(code: int, message: str = "") -> str:
    path = Path(tempfile.mkdtemp(prefix="thermal_fake_")) / "gate.sh"
    path.write_text("#!/bin/bash\n" + (f'echo "{message}"\n' if message else "") + f"exit {code}\n")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


class ExternalCommandContract(unittest.TestCase):
    """The same 0/1/2 contract a hand-written gate script follows."""

    def gate(self, command: str) -> ThermalGate:
        return ThermalGate(ThermalConfig(command=command), prober=FakeHost().hot(), env={})

    def test_exit0_passes_for_all_profiles(self):
        g = self.gate(_script(0))
        for p in ALL:
            with self.subTest(profile=p):
                g.enforce(p)

    def test_exit1_on_medium_is_a_downgrade(self):
        with self.assertRaises(ThermalDowngrade):
            self.gate(_script(1, "DOWNGRADE")).enforce("medium")

    def test_exit1_on_other_profiles_is_a_hard_block(self):
        for p in ("standard", "coder", "heavy"):
            with self.subTest(profile=p):
                with self.assertRaises(ThermalBlocked):
                    self.gate(_script(1, "YELLOW")).enforce(p)

    def test_exit2_blocks_everything_even_medium(self):
        for p in ALL:
            with self.subTest(profile=p):
                with self.assertRaises(ThermalBlocked):
                    self.gate(_script(2, "SKIP_LLM")).enforce(p)

    def test_script_message_is_surfaced(self):
        with self.assertRaises(ThermalBlocked) as ctx:
            self.gate(_script(2, "SKIP_LLM")).enforce("light")
        self.assertIn("SKIP_LLM", str(ctx.exception))

    def test_missing_script_fails_open(self):
        self.gate("/nonexistent/path/gate.sh").enforce("heavy")

    def test_hanging_script_fails_closed(self):
        def hang(*a, **k):
            raise subprocess.TimeoutExpired(cmd="gate", timeout=3)
        g = ThermalGate(ThermalConfig(command=_script(0)), run=hang, env={})
        with self.assertRaises(ThermalBlocked):
            g.enforce("standard")

    def test_unspawnable_script_fails_open(self):
        def boom(*a, **k):
            raise OSError("bad interpreter")
        g = ThermalGate(ThermalConfig(command=_script(0)), run=boom, env={})
        g.enforce("standard")


class RamPressureDemotion(unittest.TestCase):
    """Heavy-tier demotion: low free RAM, or another heavy session running."""

    VM_AMPLE = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                            4000000.\n"
                "Pages inactive:                         100000.\n")
    VM_TIGHT = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\n"
                "Pages free:                             500000.\n"
                "Pages inactive:                          50000.\n")

    @staticmethod
    def fake_run(pgrep_out: str, vm_out: str):
        class R:
            def __init__(self, stdout):
                self.stdout, self.returncode, self.stderr = stdout, 0, ""

        def run(cmd, *a, **k):
            return R(pgrep_out if cmd[0] == "pgrep" else vm_out if cmd[0] == "vm_stat" else "")
        return run

    def demote(self, pgrep, vm, **cfg):
        c = ThermalConfig(**cfg)
        return should_demote_heavy(c, self.fake_run(pgrep, vm), platform="darwin")

    def test_ample_ram_no_session_does_not_demote(self):
        self.assertFalse(self.demote("", self.VM_AMPLE))

    def test_low_ram_demotes(self):
        self.assertTrue(self.demote("", self.VM_TIGHT))          # ~8.4 GiB free < 22

    def test_threshold_is_configurable(self):
        self.assertFalse(self.demote("", self.VM_TIGHT, min_free_gb_heavy=4.0))

    def test_concurrent_session_demotes_when_pattern_configured(self):
        self.assertTrue(self.demote("12345 some-heavy-job\n", self.VM_AMPLE,
                                    pressure_process_pattern="some-heavy-job"))

    def test_pattern_is_off_by_default(self):
        self.assertFalse(self.demote("12345 some-heavy-job\n", self.VM_AMPLE))

    def test_always_on_daemons_are_ignored(self):
        # A permanent background daemon whose command line merely contains the
        # pattern must not demote the heavy tier forever.
        out = "1378 python3 /opt/tools/job_config_guard.py\n999 bash logs-trim.sh\n"
        self.assertFalse(self.demote(out, self.VM_AMPLE, pressure_process_pattern="job|logs",
                                     pressure_ignore=["config_guard", "logs-trim"]))

    def test_unreadable_memory_does_not_demote(self):
        self.assertFalse(self.demote("", ""))

    def test_vm_stat_parsing(self):
        self.assertAlmostEqual(free_memory_gb(self.fake_run("", self.VM_AMPLE), "darwin"),
                               (4_100_000 * 16384) / 1024 ** 3, places=3)
        self.assertGreater(_free_pct_from_vm_stat(
            "Pages free: 100.\nPages active: 100.\nPages inactive: 100.\nPages wired down: 100.\n"), 24)


if __name__ == "__main__":
    unittest.main(verbosity=2)

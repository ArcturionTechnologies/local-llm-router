"""The shipped examples must stay valid as the schema evolves."""

import json
import subprocess
import unittest
from pathlib import Path

from local_llm_router.config import load_config
from local_llm_router.router import Router, pick_tier

EX = Path(__file__).resolve().parent.parent / "examples"
ENV = {"LLM_ROUTER_HOME": "/nonexistent-home"}


class ExampleTests(unittest.TestCase):
    def test_lmstudio_example_routes_every_local_role_to_one_server(self):
        cfg = load_config(EX / "lmstudio.toml", env=ENV)
        for tc in ("classify", "code-gen", "reason", "vision"):
            self.assertEqual(pick_tier(cfg, task_class=tc)[0], "lmstudio")
        self.assertEqual(cfg.tier("lmstudio").base_url, "http://127.0.0.1:1234/v1")
        chain = [t.name for t in Router(cfg).chain("lmstudio")]
        self.assertEqual(chain, ["lmstudio", "groq", "cerebras", "gemini"])

    def test_mlx_multi_server_example_loads(self):
        cfg = load_config(EX / "mlx-multi-server.toml", env=ENV)
        self.assertTrue(cfg.tier("heavy").exclusive)
        self.assertEqual(cfg.thermal.pressure_process_pattern, "claude|codex")

    def test_heartbeat_manifest_is_valid_json_with_known_keys(self):
        doc = json.loads((EX / "heartbeat-signals" / "inbox.json").read_text())
        self.assertLessEqual(set(doc), {"files", "dirs", "commands", "sanity_fire_hours",
                                        "skip_when_hot", "log_path"})

    def test_thermal_gate_script_is_valid_bash(self):
        self.assertEqual(subprocess.run(["bash", "-n", str(EX / "thermal-gate-script.sh")]).returncode, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""Config: defaults, TOML/JSON overlay, environment overrides, strict validation."""

import json
import tempfile
import tomllib
import unittest
from pathlib import Path

from local_llm_router.config import Config, apply_dict, apply_env, dump_toml, load_config
from local_llm_router.errors import ConfigError


class Defaults(unittest.TestCase):
    def test_defaults_validate(self):
        Config().validate()

    def test_every_fallback_role_and_complexity_names_a_real_tier(self):
        cfg = Config()
        for t in cfg.tiers.values():
            for fb in t.fallbacks:
                self.assertIn(fb, cfg.tiers)
        self.assertTrue(set(cfg.roles.values()) <= set(cfg.tiers))
        self.assertTrue(set(cfg.complexity_tiers.values()) <= set(cfg.tiers))

    def test_no_default_url_leaves_the_machine_except_cloud_apis(self):
        for t in Config().tiers.values():
            if t.kind == "local":
                self.assertTrue(t.base_url.startswith("http://127.0.0.1:"), t.name)

    def test_no_default_contains_a_secret(self):
        blob = json.dumps(Config().to_dict())
        self.assertNotIn("sk-", blob)
        for t in Config().tiers.values():
            if t.kind == "cloud":
                self.assertTrue(t.api_key_env.endswith("_KEY"))   # keys are env var *names* only

    def test_profile_and_cost_defaults_follow_kind(self):
        cfg = Config()
        self.assertEqual((cfg.tier("fast").profile, cfg.tier("fast").cost_class), ("light", "local"))
        self.assertEqual((cfg.tier("sonnet").profile, cfg.tier("sonnet").cost_class), ("none", "paid"))


class Overlay(unittest.TestCase):
    def test_partial_tier_override_keeps_the_other_fields(self):
        cfg = apply_dict(Config(), {"tiers": {"fast": {"model": "my-4b", "timeout": 3}}})
        self.assertEqual((cfg.tier("fast").model, cfg.tier("fast").timeout), ("my-4b", 3.0))
        self.assertEqual(cfg.tier("fast").thermal, "light")

    def test_new_tier_can_be_added_and_used_as_a_fallback(self):
        cfg = apply_dict(Config(), {"tiers": {
            "lmstudio": {"base_url": "http://127.0.0.1:1234/v1", "model": "qwen3-8b", "fallbacks": ["groq"]},
            "fast": {"fallbacks": ["lmstudio", "groq"]}}})
        cfg.validate()
        self.assertEqual(cfg.tier("lmstudio").profile, "standard")

    def test_sections_and_dict_merging(self):
        cfg = apply_dict(Config(), {"thermal": {"mem_low_pct": 25}, "roles": {"fast": "small"},
                                    "budget": {"enforce": True}})
        self.assertEqual(cfg.thermal.mem_low_pct, 25.0)
        self.assertEqual(cfg.roles["fast"], "small")
        self.assertEqual(cfg.roles["medium"], "medium")            # untouched
        self.assertTrue(cfg.budget.enforce)

    def test_unknown_keys_are_rejected_at_every_level(self):
        for doc in ({"nope": 1}, {"thermal": {"nope": 1}}, {"tiers": {"fast": {"nope": 1}}}):
            with self.subTest(doc=doc):
                with self.assertRaises(ConfigError):
                    apply_dict(Config(), doc)

    def test_type_errors_are_rejected(self):
        for doc in ({"thermal": {"enabled": "yes"}}, {"thermal": {"mem_low_pct": "x"}},
                    {"tiers": {"fast": {"fallbacks": "small"}}}, {"routing": {"bulk_threshold": 1.5}}):
            with self.subTest(doc=doc):
                with self.assertRaises(ConfigError):
                    apply_dict(Config(), doc)

    def test_validation_catches_dangling_references_and_bad_enums(self):
        for doc in ({"tiers": {"fast": {"fallbacks": ["ghost"]}}},
                    {"roles": {"fast": "ghost"}},
                    {"complexity_tiers": {"hard": "ghost"}},
                    {"tiers": {"fast": {"thermal": "scorching"}}},
                    {"tiers": {"fast": {"protocol": "carrier-pigeon"}}},
                    {"tiers": {"fresh": {"kind": "local"}}}):          # new tier without base_url
            with self.subTest(doc=doc):
                with self.assertRaises(ConfigError):
                    apply_dict(Config(), doc).validate()


class Files(unittest.TestCase):
    def write(self, name, text):
        d = Path(tempfile.mkdtemp(prefix="llmrouter_cfg_"))
        (d / name).write_text(text)
        return d

    def test_toml_file_is_found_in_the_home_dir(self):
        home = self.write("config.toml", '[tiers.fast]\nmodel = "from-file"\n')
        cfg = load_config(env={"LLM_ROUTER_HOME": str(home)})
        self.assertEqual(cfg.tier("fast").model, "from-file")
        self.assertEqual(cfg.home, home)

    def test_json_file_works_too(self):
        home = self.write("config.json", json.dumps({"tiers": {"fast": {"model": "from-json"}}}))
        self.assertEqual(load_config(env={"LLM_ROUTER_HOME": str(home)}).tier("fast").model, "from-json")

    def test_explicit_config_env_beats_the_home_dir(self):
        home = self.write("config.toml", '[tiers.fast]\nmodel = "home"\n')
        other = self.write("x.toml", '[tiers.fast]\nmodel = "explicit"\n')
        cfg = load_config(env={"LLM_ROUTER_HOME": str(home), "LLM_ROUTER_CONFIG": str(other / "x.toml")})
        self.assertEqual(cfg.tier("fast").model, "explicit")

    def test_no_file_means_defaults(self):
        home = Path(tempfile.mkdtemp(prefix="llmrouter_cfg_"))
        self.assertEqual(load_config(env={"LLM_ROUTER_HOME": str(home)}).tier("fast").thermal, "light")

    def test_missing_explicit_file_and_bad_syntax_are_config_errors(self):
        with self.assertRaises(ConfigError):
            load_config("/nonexistent/config.toml", env={})
        home = self.write("config.toml", "this is = = not toml")
        with self.assertRaises(ConfigError):
            load_config(env={"LLM_ROUTER_HOME": str(home)})

    def test_state_lives_under_home_by_default(self):
        home = Path(tempfile.mkdtemp(prefix="llmrouter_cfg_"))
        cfg = load_config(env={"LLM_ROUTER_HOME": str(home)})
        self.assertEqual(cfg.state_path, home / "state")
        self.assertEqual(cfg.telemetry_path, home / "state" / "telemetry.jsonl")
        cfg = load_config(env={"LLM_ROUTER_HOME": str(home), "LLM_ROUTER_STATE_DIR": str(home / "elsewhere")})
        self.assertEqual(cfg.state_path, home / "elsewhere")


class Environment(unittest.TestCase):
    def test_one_variable_retargets_every_local_tier_but_no_cloud_tier(self):
        cfg = apply_env(Config(), {"LLM_ROUTER_LOCAL_BASE_URL": "http://127.0.0.1:1234/v1",
                                   "LLM_ROUTER_LOCAL_MODEL": "qwen3-8b"})
        for t in cfg.tiers.values():
            if t.kind == "local":
                self.assertEqual((t.base_url, t.model), ("http://127.0.0.1:1234/v1", "qwen3-8b"))
        self.assertEqual(cfg.tier("groq").base_url, "https://api.groq.com/openai/v1")

    def test_per_tier_overrides_win_over_the_global_one(self):
        cfg = apply_env(Config(), {"LLM_ROUTER_LOCAL_BASE_URL": "http://127.0.0.1:1234/v1",
                                   "LLM_ROUTER_URL_HEAVY": "http://127.0.0.1:9999/v1",
                                   "LLM_ROUTER_MODEL_GROQ": "other-model"})
        self.assertEqual(cfg.tier("heavy").base_url, "http://127.0.0.1:9999/v1")
        self.assertEqual(cfg.tier("fast").base_url, "http://127.0.0.1:1234/v1")
        self.assertEqual(cfg.tier("groq").model, "other-model")


class TomlRoundTrip(unittest.TestCase):
    def test_dump_then_load_reproduces_the_defaults(self):
        def prune(x):          # empty tables are intentionally not written
            return {k: prune(v) for k, v in x.items() if v != {}} if isinstance(x, dict) else x
        d = Config().to_dict()
        self.assertEqual(tomllib.loads(dump_toml(d)), prune(d))

    def test_written_file_loads_back_into_an_equal_config(self):
        home = Path(tempfile.mkdtemp(prefix="llmrouter_cfg_"))
        (home / "config.toml").write_text(dump_toml(Config().to_dict()))
        self.assertEqual(load_config(env={"LLM_ROUTER_HOME": str(home)}).to_dict(), Config().to_dict())

    def test_extra_payload_survives_the_round_trip(self):
        cfg = apply_dict(Config(), {"tiers": {"fast": {"extra_payload": {"enable_thinking": False}}}})
        back = apply_dict(Config(), tomllib.loads(dump_toml(cfg.to_dict())))
        self.assertEqual(back.tier("fast").extra_payload, {"enable_thinking": False})


if __name__ == "__main__":
    unittest.main(verbosity=2)

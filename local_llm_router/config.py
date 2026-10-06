"""Configuration: defaults, TOML/JSON loading, environment overrides.

Resolution order (later wins):

1. Built-in defaults (:func:`default_tiers` and the section dataclasses).
2. ``config.toml`` or ``config.json`` -- ``$LLM_ROUTER_CONFIG`` if set, else
   ``$LLM_ROUTER_HOME/config.toml`` (default home: ``~/.config/local-llm-router``).
3. Environment overrides (see :func:`apply_env`).

Unknown keys are an error, so a typo fails loudly instead of silently doing
nothing.
"""

from __future__ import annotations

import dataclasses
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Optional

try:  # Python 3.11+
    import tomllib
except ImportError:  # pragma: no cover - older interpreters use config.json
    tomllib = None  # type: ignore[assignment]

from .errors import ConfigError

ENV_HOME = "LLM_ROUTER_HOME"
ENV_CONFIG = "LLM_ROUTER_CONFIG"
ENV_STATE = "LLM_ROUTER_STATE_DIR"

PROFILES = ("none", "light", "standard", "medium", "coder", "vision", "heavy")
PROTOCOLS = ("openai", "anthropic")
KINDS = ("local", "cloud")
COSTS = ("local", "free", "paid")


def home_dir(env: Optional[Mapping[str, str]] = None) -> Path:
    env = os.environ if env is None else env
    return Path(env.get(ENV_HOME) or "~/.config/local-llm-router").expanduser()


# --------------------------------------------------------------------------- #
# Section dataclasses
# --------------------------------------------------------------------------- #

@dataclass
class Tier:
    """One routable backend: a local server or a cloud API."""

    name: str
    kind: str = "local"              # "local" | "cloud"
    protocol: str = "openai"         # "openai" (chat/completions) | "anthropic" (messages)
    base_url: str = ""               # OpenAI: ".../v1"   Anthropic: "https://api.anthropic.com"
    model: str = ""
    api_key_env: str = ""            # env var that holds the key (cloud tiers)
    timeout: float = 30.0
    thermal: str = ""                # profile: none|light|standard|medium|coder|vision|heavy
    exclusive: bool = False          # hold the model lock around each call
    server: str = ""                 # lock-holder name (defaults to the tier name)
    demote_on_pressure: bool = False  # skip when RAM is tight / another heavy session runs
    fallbacks: list = field(default_factory=list)
    cost: str = ""                   # local | free | paid (defaults from kind)
    price_in_per_mtok: float = 0.0   # USD per million input tokens (paid tiers)
    price_out_per_mtok: float = 0.0  # USD per million output tokens (paid tiers)
    daily_token_limit: int = 0       # informational ceiling for free tiers (0 = unknown)
    context_window: int = 32768
    enabled: bool = True
    extra_payload: dict = field(default_factory=dict)  # merged into the request body

    @property
    def profile(self) -> str:
        return self.thermal or ("none" if self.kind == "cloud" else "standard")

    @property
    def cost_class(self) -> str:
        return self.cost or ("local" if self.kind == "local" else "paid")

    @property
    def lock_name(self) -> str:
        return self.server or self.name

    def validate(self) -> None:
        if self.kind not in KINDS:
            raise ConfigError(f"tiers.{self.name}.kind: {self.kind!r} not in {KINDS}")
        if self.protocol not in PROTOCOLS:
            raise ConfigError(f"tiers.{self.name}.protocol: {self.protocol!r} not in {PROTOCOLS}")
        if self.profile not in PROFILES:
            raise ConfigError(f"tiers.{self.name}.thermal: {self.profile!r} not in {PROFILES}")
        if self.cost_class not in COSTS:
            raise ConfigError(f"tiers.{self.name}.cost: {self.cost_class!r} not in {COSTS}")
        if not self.base_url:
            raise ConfigError(f"tiers.{self.name}.base_url is required")


@dataclass
class RoutingConfig:
    bulk_threshold: int = 20            # strictly greater than this => bulk loop
    long_context_tokens: int = 100_000  # strictly greater than this => long-context tier


@dataclass
class ThermalConfig:
    enabled: bool = True
    command: str = ""                   # optional external gate: "<command> <profile>", exit 0/1/2
    bypass_env: str = "LLM_ROUTER_BYPASS_THERMAL"  # set to "1" to skip the gate (lean/cron jobs)
    cpu_heavy_ratio: float = 0.8        # load1 > ratio * cores  => red
    cpu_moderate_ratio: float = 0.5     # load1 > ratio * cores  => yellow
    mem_low_pct: float = 15.0           # free memory below this => yellow
    mem_critical_pct: float = 8.0       # free memory below this => red
    battery_low_pct: int = 15           # on battery and below this => red
    throttle_speed_limit: int = 90      # pmset CPU_Speed_Limit below this => red
    min_free_gb_heavy: float = 22.0     # heavy tiers demote below this much free RAM
    pressure_process_pattern: str = ""  # pgrep -f regex; a match demotes heavy tiers
    pressure_ignore: list = field(default_factory=list)  # substrings to ignore in matches


@dataclass
class BudgetConfig:
    paid_monthly_usd: float = 100.0     # ceiling across all paid tiers
    warn_pct: float = 80.0
    block_pct: float = 100.0
    enforce: bool = False               # True => refuse paid tiers once the envelope is spent


@dataclass
class RateGuardConfig:
    max_concurrent: int = 3             # simultaneous spawned sessions
    max_1h: int = 5                     # spawns in a rolling hour
    max_5h: int = 8                     # spawns in a rolling 5 hours
    max_week: int = 50                  # spawns in a rolling week
    cooldown_min: int = 60              # back-off after an explicit rate-limit signal
    yellow_pct: float = 0.80
    red_pct: float = 0.95
    process_pattern: str = "claude -p"  # pgrep -f pattern that counts active sessions
    rate_limit_regex: str = r"rate.?limit|overloaded|too many requests|\b529\b|quota"
    spawn_log_dir: str = ""             # directory of *.log files scanned for rate-limit signals
    registry_path: str = ""             # optional extra JSONL of session_start events
    kill_flags: list = field(default_factory=list)  # extra kill-switch files (state/KILL_ACTIVE is always checked)


@dataclass
class LocalFirstConfig:
    paid_providers: list = field(default_factory=lambda: [
        "claude", "claude-opus", "claude-sonnet", "claude-haiku",
        "anthropic", "codex", "gpt-5", "gpt-4", "openai"])
    exempt_task_classes: list = field(default_factory=lambda: [
        "irreversible", "money-transfer", "external-send", "prod-deploy",
        "schema-change", "security-audit", "legal-correspondence"])
    shell_tools: list = field(default_factory=lambda: [
        "Bash", "shell", "unified_exec", "exec_command", "functions.exec_command"])
    paid_markers: list = field(default_factory=lambda: [
        "claude -p ", "codex -p ", "claude --prompt",
        "anthropic.messages.create", "openai.chat.completions"])


@dataclass
class TelemetryConfig:
    enabled: bool = True
    path: str = ""                      # default: <state_dir>/telemetry.jsonl


@dataclass
class HeartbeatConfig:
    signals_dir: str = ""               # default: <home>/heartbeat/signals
    state_dir: str = ""                 # default: <state_dir>/heartbeat
    tasks_dir: str = ""                 # directory whose sub-folders are heartbeat ids (for `discover`)
    sanity_fire_hours: float = 6.0


DEFAULT_ROLES = {
    "fast": "fast",              # snap classify / route / trivial summary
    "helper": "small",           # tier used by classify/extract/summarize helpers
    "small": "small",
    "medium": "medium",
    "coder": "coder",
    "heavy": "heavy",
    "vision": "vision",
    "bulk_simple": "groq",
    "bulk_complex": "cerebras",
    "long_context": "gemini",
    "strong": "sonnet",
    "strongest": "opus",
}

DEFAULT_COMPLEXITY_TIERS = {
    "trivial": "fast",
    "routine": "medium",
    "hard": "sonnet",
    "critical": "opus",
}


def default_tiers() -> dict:
    """Six local tiers (one per workload) and six cloud tiers (free then paid).

    Every field is overridable from config. A single-server setup (LM Studio,
    one ``mlx_lm.server``) just points several tiers at the same URL, or sets
    ``LLM_ROUTER_LOCAL_BASE_URL``.
    """
    local = "mlx-community/"
    return {t.name: t for t in [
        Tier("fast", base_url="http://127.0.0.1:8770/v1", model=local + "Qwen3-4B-Instruct-2507-4bit",
             thermal="light", timeout=8.0, fallbacks=["small", "groq"]),
        Tier("small", base_url="http://127.0.0.1:8765/v1", model=local + "Qwen3-8B-4bit",
             thermal="standard", timeout=15.0, fallbacks=["groq", "cerebras"]),
        Tier("medium", base_url="http://127.0.0.1:8766/v1", model=local + "Qwen3-14B-4bit",
             thermal="medium", timeout=30.0, fallbacks=["small", "groq"]),
        Tier("coder", base_url="http://127.0.0.1:8772/v1", model=local + "Qwen2.5-Coder-14B-Instruct-4bit",
             thermal="coder", timeout=60.0, fallbacks=["medium", "groq"]),
        Tier("heavy", base_url="http://127.0.0.1:8768/v1", model=local + "Qwen3-30B-A3B-Instruct-2507-4bit",
             thermal="heavy", timeout=90.0, exclusive=True, demote_on_pressure=True,
             fallbacks=["medium", "groq"]),
        Tier("vision", base_url="http://127.0.0.1:8769/v1", model=local + "Qwen2.5-VL-7B-Instruct-4bit",
             thermal="vision", timeout=45.0, fallbacks=[]),
        Tier("groq", kind="cloud", base_url="https://api.groq.com/openai/v1",
             model="llama-3.3-70b-versatile", api_key_env="GROQ_API_KEY", cost="free",
             timeout=30.0, daily_token_limit=14_400_000, context_window=128_000,
             fallbacks=["cerebras", "gemini"]),
        Tier("cerebras", kind="cloud", base_url="https://api.cerebras.ai/v1",
             model="gpt-oss-120b", api_key_env="CEREBRAS_API_KEY", cost="free",
             timeout=30.0, daily_token_limit=1_000_000, context_window=128_000,
             fallbacks=["gemini"]),
        Tier("gemini", kind="cloud", base_url="https://generativelanguage.googleapis.com/v1beta/openai",
             model="gemini-2.5-flash", api_key_env="GEMINI_API_KEY", cost="free",
             timeout=60.0, context_window=1_000_000, fallbacks=[]),
        Tier("haiku", kind="cloud", protocol="anthropic", base_url="https://api.anthropic.com",
             model="claude-haiku-4-5-20251001", api_key_env="ANTHROPIC_API_KEY", cost="paid",
             timeout=60.0, price_in_per_mtok=0.80, price_out_per_mtok=4.0,
             context_window=200_000, fallbacks=["sonnet"]),
        Tier("sonnet", kind="cloud", protocol="anthropic", base_url="https://api.anthropic.com",
             model="claude-sonnet-5-5", api_key_env="ANTHROPIC_API_KEY", cost="paid",
             timeout=90.0, price_in_per_mtok=3.0, price_out_per_mtok=15.0,
             context_window=200_000, fallbacks=["opus"]),
        Tier("opus", kind="cloud", protocol="anthropic", base_url="https://api.anthropic.com",
             model="claude-opus-5-5", api_key_env="ANTHROPIC_API_KEY", cost="paid",
             timeout=120.0, price_in_per_mtok=15.0, price_out_per_mtok=75.0,
             context_window=200_000, fallbacks=[]),
    ]}


@dataclass
class Config:
    tiers: dict = field(default_factory=default_tiers)
    roles: dict = field(default_factory=lambda: dict(DEFAULT_ROLES))
    complexity_tiers: dict = field(default_factory=lambda: dict(DEFAULT_COMPLEXITY_TIERS))
    routing: RoutingConfig = field(default_factory=RoutingConfig)
    thermal: ThermalConfig = field(default_factory=ThermalConfig)
    budget: BudgetConfig = field(default_factory=BudgetConfig)
    rate_guard: RateGuardConfig = field(default_factory=RateGuardConfig)
    local_first: LocalFirstConfig = field(default_factory=LocalFirstConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    heartbeat: HeartbeatConfig = field(default_factory=HeartbeatConfig)
    state_dir: str = ""
    home: Path = field(default_factory=home_dir)

    # -- derived paths ------------------------------------------------------
    @property
    def state_path(self) -> Path:
        return Path(self.state_dir).expanduser() if self.state_dir else self.home / "state"

    @property
    def telemetry_path(self) -> Path:
        return Path(self.telemetry.path).expanduser() if self.telemetry.path else self.state_path / "telemetry.jsonl"

    @property
    def lock_dir(self) -> Path:
        return self.state_path / "locks"

    # -- helpers ------------------------------------------------------------
    def tier(self, name: str) -> Tier:
        try:
            return self.tiers[name]
        except KeyError:
            raise ConfigError(f"unknown tier {name!r} (known: {', '.join(sorted(self.tiers))})") from None

    def role(self, role: str) -> str:
        return self.roles.get(role, role)

    def validate(self) -> "Config":
        for t in self.tiers.values():
            t.validate()
            for fb in t.fallbacks:
                if fb not in self.tiers:
                    raise ConfigError(f"tiers.{t.name}.fallbacks: unknown tier {fb!r}")
        for role, name in self.roles.items():
            if name not in self.tiers:
                raise ConfigError(f"roles.{role}: unknown tier {name!r}")
        for cx, name in self.complexity_tiers.items():
            if name not in self.tiers:
                raise ConfigError(f"complexity_tiers.{cx}: unknown tier {name!r}")
        return self

    def to_dict(self) -> dict:
        def conv(v: Any) -> Any:
            if dataclasses.is_dataclass(v) and not isinstance(v, type):
                return {f.name: conv(getattr(v, f.name)) for f in dataclasses.fields(v)}
            if isinstance(v, dict):
                return {k: conv(x) for k, x in v.items()}
            if isinstance(v, (list, tuple)):
                return [conv(x) for x in v]
            if isinstance(v, Path):
                return str(v)
            return v
        d = conv(self)
        d.pop("home", None)
        if not d.get("state_dir"):
            d.pop("state_dir", None)
        return d


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _coerce(current: Any, value: Any, path: str) -> Any:
    if isinstance(current, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if isinstance(current, int):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or int(value) != value:
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return int(value)
    if isinstance(current, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if isinstance(current, str):
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    if isinstance(current, list):
        if not isinstance(value, list):
            raise ConfigError(f"{path}: expected a list, got {value!r}")
        return list(value)
    if isinstance(current, dict):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a table, got {value!r}")
        return dict(value)
    return value


def _apply(obj: Any, data: Any, path: str) -> None:
    if not isinstance(data, dict):
        raise ConfigError(f"{path}: expected a table, got {data!r}")
    known = {f.name for f in dataclasses.fields(obj)}
    for key, value in data.items():
        if key not in known:
            raise ConfigError(f"{path}.{key}: unknown key (valid: {', '.join(sorted(known))})")
        current = getattr(obj, key)
        sub = f"{path}.{key}"
        if dataclasses.is_dataclass(current):
            _apply(current, value, sub)
        elif isinstance(current, dict):
            if not isinstance(value, dict):
                raise ConfigError(f"{sub}: expected a table, got {value!r}")
            current.update(value)           # merge: a user can remap one role/complexity
        else:
            setattr(obj, key, _coerce(current, value, sub))


def _apply_tiers(cfg: Config, tiers: Any) -> None:
    if not isinstance(tiers, dict):
        raise ConfigError("tiers: expected a table of tables")
    for name, data in tiers.items():
        if not isinstance(data, dict):
            raise ConfigError(f"tiers.{name}: expected a table")
        tier = cfg.tiers.get(name)
        if tier is None:
            tier = Tier(name=name)
            cfg.tiers[name] = tier
        data = {k: v for k, v in data.items() if k != "name"}
        _apply(tier, data, f"tiers.{name}")


def apply_dict(cfg: Config, data: Mapping[str, Any]) -> Config:
    """Overlay a parsed config document onto ``cfg`` (mutates and returns it)."""
    data = dict(data)
    if "tiers" in data:
        _apply_tiers(cfg, data.pop("tiers"))
    _apply(cfg, data, "config")
    return cfg


def apply_env(cfg: Config, env: Optional[Mapping[str, str]] = None) -> Config:
    """Environment overrides.

    ``LLM_ROUTER_LOCAL_BASE_URL`` / ``LLM_ROUTER_LOCAL_MODEL`` retarget every
    *local* tier at one server (LM Studio, a single ``mlx_lm.server``).
    ``LLM_ROUTER_URL_<TIER>`` and ``LLM_ROUTER_MODEL_<TIER>`` retarget one tier
    (tier name upper-cased, ``-`` as ``_``). ``LLM_ROUTER_STATE_DIR`` moves state.
    """
    env = os.environ if env is None else env
    local_url, local_model = env.get("LLM_ROUTER_LOCAL_BASE_URL"), env.get("LLM_ROUTER_LOCAL_MODEL")
    for tier in cfg.tiers.values():
        if tier.kind == "local":
            if local_url:
                tier.base_url = local_url
            if local_model:
                tier.model = local_model
        key = re.sub(r"[^A-Z0-9]", "_", tier.name.upper())
        if env.get(f"LLM_ROUTER_URL_{key}"):
            tier.base_url = env[f"LLM_ROUTER_URL_{key}"]
        if env.get(f"LLM_ROUTER_MODEL_{key}"):
            tier.model = env[f"LLM_ROUTER_MODEL_{key}"]
    if env.get(ENV_STATE):
        cfg.state_dir = env[ENV_STATE]
    return cfg


def find_config_file(env: Optional[Mapping[str, str]] = None) -> Optional[Path]:
    env = os.environ if env is None else env
    explicit = env.get(ENV_CONFIG)
    if explicit:
        return Path(explicit).expanduser()
    home = home_dir(env)
    for name in ("config.toml", "config.json"):
        if (home / name).is_file():
            return home / name
    return None


def parse_file(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise ConfigError(f"{path}: invalid JSON: {e}") from e
    if tomllib is None:
        raise ConfigError(f"{path}: TOML needs Python 3.11+; use config.json instead")
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"{path}: invalid TOML: {e}") from e


def load_config(path: Optional[os.PathLike] = None,
                env: Optional[Mapping[str, str]] = None) -> Config:
    """Build the effective :class:`Config` (defaults + file + environment)."""
    env = os.environ if env is None else env
    cfg = Config(home=home_dir(env))
    file = Path(path).expanduser() if path else find_config_file(env)
    if file is not None:
        if not file.is_file():
            raise ConfigError(f"config file not found: {file}")
        apply_dict(cfg, parse_file(file))
    apply_env(cfg, env)
    return cfg.validate()


# --------------------------------------------------------------------------- #
# Minimal TOML writer (for `llm-router config init`)
# --------------------------------------------------------------------------- #

_BARE = re.compile(r"^[A-Za-z0-9_-]+$")


def _key(k: str) -> str:
    return k if _BARE.match(k) else json.dumps(k)


def _val(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, str):
        return json.dumps(v)
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_val(x) for x in v) + "]"
    raise TypeError(f"cannot write {type(v).__name__} as TOML")


def dump_toml(data: Mapping[str, Any], _prefix: str = "") -> str:
    """Serialise nested dicts of scalars/lists to TOML (empty tables are omitted)."""
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    tables = {k: v for k, v in data.items() if isinstance(v, dict) and v}
    out = [f"{_key(k)} = {_val(v)}" for k, v in scalars.items()]
    for k, v in tables.items():
        name = f"{_prefix}.{_key(k)}" if _prefix else _key(k)
        body = dump_toml(v, name)
        if any(not isinstance(x, dict) for x in v.values()):
            out.append(f"\n[{name}]")
        out.append(body.rstrip("\n"))
    return "\n".join(line for line in out if line is not None).strip("\n") + "\n"

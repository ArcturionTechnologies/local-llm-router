"""local-llm-router: a local-first LLM router for Apple Silicon.

Send work to on-device (MLX / OpenAI-compatible) servers first; escalate to a
bigger tier when a task needs it; back off when the Mac runs hot or busy.

    from local_llm_router import chat, classify, pick_tier, Router

    label = classify("AAPL beats EPS", ["earnings", "macro", "noise"])
    reply = Router().complete("Plan a migration", task_class="reason", complexity="complex")
"""

from __future__ import annotations

from .budget import Budget
from .config import Config, Tier, dump_toml, load_config
from .errors import (AllTiersFailed, BudgetExceeded, ConfigError, MissingCredentials,
                     RateLimited, RouterError, ThermalBlocked, ThermalDowngrade, TierBusy,
                     TierError)
from .heartbeat import HeartbeatGate
from .local_first import LocalFirstGate
from .model_lock import ModelLock
from .rate_guard import RateGuard
from .router import (Attempt, Completion, Router, get_router, pick_tier, set_router)
from .thermal import Decision, Snapshot, ThermalGate

__version__ = "0.1.0"


# -- module-level conveniences backed by the default router ---------------------

def chat(prompt, **kw) -> str:
    return get_router().chat(prompt, **kw)


def complete(prompt, **kw) -> Completion:
    return get_router().complete(prompt, **kw)


def classify(text, labels, **kw) -> str:
    return get_router().classify(text, labels, **kw)


def extract_json(text, schema_hint, **kw) -> dict:
    return get_router().extract_json(text, schema_hint, **kw)


def summarize(text, max_words: int = 40, **kw) -> str:
    return get_router().summarize(text, max_words, **kw)


def route(prompt, options, **kw) -> str:
    return get_router().route(prompt, options, **kw)


def should_escalate(prompt, **kw) -> bool:
    return get_router().should_escalate(prompt, **kw)


def tier_call(prompt, **kw) -> Completion:
    return get_router().tier_call(prompt, **kw)


def health(**kw) -> dict:
    return get_router().health(**kw)


def pick(task_class="other", complexity="simple", stakes="low", bulk_count=1, context_tokens=0):
    return get_router().pick(task_class, complexity, stakes, bulk_count, context_tokens)


__all__ = [
    "AllTiersFailed", "Attempt", "Budget", "BudgetExceeded", "Completion", "Config",
    "ConfigError", "Decision", "HeartbeatGate", "LocalFirstGate", "MissingCredentials",
    "ModelLock", "RateGuard", "RateLimited", "Router", "RouterError", "Snapshot",
    "ThermalBlocked", "ThermalDowngrade", "ThermalGate", "Tier", "TierBusy", "TierError",
    "chat", "classify", "complete", "dump_toml", "extract_json", "get_router", "health",
    "load_config", "pick", "pick_tier", "route", "set_router", "should_escalate",
    "summarize", "tier_call",
]

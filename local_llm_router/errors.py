"""Exception hierarchy.

Everything the router can raise derives from :class:`RouterError`, so a caller
that only wants "did the router fail?" can catch one type. The subclasses let
you tell *why* a tier was skipped.
"""

from __future__ import annotations

from typing import Optional


class RouterError(RuntimeError):
    """Base class for every error raised by this package."""


class ConfigError(RouterError):
    """The configuration file or an environment override is invalid."""


class TierError(RouterError):
    """A tier could not serve the request (transport, HTTP or malformed reply)."""


class TierBusy(TierError):
    """The tier is healthy but unavailable right now (model lock held elsewhere).

    Servers that follow the ``local-pool-busy`` contract answer HTTP 503 with
    ``{"error": "local-pool-busy", "holder": ..., "holder_pid": ...}``. A 503
    without that body is a genuine fault and stays a plain :class:`TierError`.
    """

    def __init__(self, message: str, holder: Optional[str] = None,
                 holder_pid: Optional[int] = None):
        super().__init__(message)
        self.holder = holder
        self.holder_pid = holder_pid


class RateLimited(TierError):
    """The provider answered 429/402, or is inside its local cool-down window."""


class MissingCredentials(TierError):
    """A cloud tier has no API key in the environment variable it names."""


class ThermalBlocked(RouterError):
    """The thermal gate refused this tier; fall through to the next one."""


class ThermalDowngrade(RouterError):
    """The thermal gate asks for a lighter tier instead of this one."""


class BudgetExceeded(RouterError):
    """A paid tier was refused because the configured envelope is spent."""


class AllTiersFailed(RouterError):
    """Every tier in the fallback chain failed, was gated, or was rejected."""

    def __init__(self, message: str, attempts: list):
        super().__init__(message)
        self.attempts = attempts

"""Append-only JSONL telemetry. Logging never breaks a call."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Mapping, Optional

_ID_RE = re.compile(r"[^a-z0-9._:-]+")


def normalize_id(value: Optional[str], prefix: str) -> str:
    """Normalise a caller-supplied id; mint a unique one when absent.

    Standalone calls get unique ids rather than a shared placeholder so they
    can never be mistaken for retries of one another.
    """
    cleaned = _ID_RE.sub("-", str(value or "").strip().lower()).strip("-.:_")
    if cleaned:
        return cleaned[:128]
    return f"{prefix}-{os.getpid():x}-{time.time_ns():x}"


def detect_caller(env: Optional[Mapping[str, str]] = None) -> str:
    """Who is calling? ``$LLM_ROUTER_CALLER`` or ``"unknown"``."""
    env = os.environ if env is None else env
    return (env.get("LLM_ROUTER_CALLER") or "unknown").strip().lower() or "unknown"


class Telemetry:
    def __init__(self, path: os.PathLike, enabled: bool = True,
                 env: Optional[Mapping[str, str]] = None):
        self.path = Path(path).expanduser()
        self.enabled = enabled
        self._env = os.environ if env is None else env

    def log(self, **record) -> None:
        if not self.enabled:
            return
        record.setdefault("ts", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        record.setdefault("caller", detect_caller(self._env))
        record.setdefault("run_id", normalize_id(self._env.get("LLM_ROUTER_RUN_ID"), "run"))
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps(record) + "\n")
        except OSError:
            pass

    def read(self) -> list:
        out = []
        try:
            with self.path.open() as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        continue
        except OSError:
            pass
        return out

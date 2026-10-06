"""Test helpers: a mock OpenAI/Anthropic-compatible HTTP server and fake host readings.

No real model is ever loaded and no real API is ever called: every "model"
below is a ``http.server`` thread on 127.0.0.1 that answers with canned JSON.
"""

from __future__ import annotations

import json
import tempfile
import threading
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from local_llm_router.config import Config
from local_llm_router.router import Router
from local_llm_router.thermal import Snapshot, ThermalGate

TEST_ENV = {"GROQ_API_KEY": "test-key", "CEREBRAS_API_KEY": "test-key",
            "GEMINI_API_KEY": "test-key", "ANTHROPIC_API_KEY": "test-key"}


def completion(text: str, tin: int = 11, tout: int = 7) -> dict:
    return {"choices": [{"message": {"role": "assistant", "content": text}}],
            "usage": {"prompt_tokens": tin, "completion_tokens": tout}}


def anthropic_reply(text: str) -> dict:
    return {"content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 5, "output_tokens": 3}}


class MockLLMServer:
    """Programmable local model server.

    ``respond(status, obj)`` queues a one-shot response for the next
    chat/messages request; with an empty queue it answers ``reply_text``.
    Every request is appended to ``requests``.
    """

    def __init__(self, reply_text: str = "ok"):
        self.reply_text = reply_text
        self.requests: list = []
        self.queue: deque = deque()
        self.models_status = 200
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):  # keep test output clean
                pass

            def _send(self, status: int, obj) -> None:
                body = obj if isinstance(obj, bytes) else json.dumps(obj).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                outer.requests.append({"method": "GET", "path": self.path, "body": None,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                if self.path.endswith("/models"):
                    self._send(outer.models_status, {"data": [{"id": "mock"}]})
                else:
                    self._send(404, {"error": "not found"})

            def do_POST(self):
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                outer.requests.append({"method": "POST", "path": self.path, "body": body,
                                       "headers": {k.lower(): v for k, v in self.headers.items()}})
                if outer.queue:
                    status, obj = outer.queue.popleft()
                    return self._send(status, obj)
                if self.path.endswith("/chat/completions"):
                    return self._send(200, completion(outer.reply_text))
                if self.path.endswith("/messages"):
                    return self._send(200, anthropic_reply(outer.reply_text))
                self._send(404, {"error": "not found"})

        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever,
                                        kwargs={"poll_interval": 0.01}, daemon=True)

    # -- lifecycle ----------------------------------------------------------
    def start(self) -> "MockLLMServer":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self._httpd.server_address[1]}"

    @property
    def v1(self) -> str:
        return self.url + "/v1"

    # -- scripting ----------------------------------------------------------
    def respond(self, status: int, obj) -> None:
        self.queue.append((status, obj))

    def posts(self) -> list:
        return [r for r in self.requests if r["method"] == "POST"]


class FakeHost:
    """A mutable host reading handed to the ThermalGate instead of probing the machine."""

    def __init__(self, **overrides):
        self.snapshot = Snapshot(cpu_load=0.5, cpu_count=10, mem_free_pct=60.0,
                                 on_battery=False, battery_pct=None, speed_limit=None,
                                 thermal_known=True)
        self.set(**overrides)

    def set(self, **kw) -> "FakeHost":
        for k, v in kw.items():
            setattr(self.snapshot, k, v)
        return self

    def hot(self) -> "FakeHost":       # red: CPU saturated
        return self.set(cpu_load=9.5)

    def busy(self) -> "FakeHost":      # yellow: CPU above 50% of cores
        return self.set(cpu_load=6.0)

    def __call__(self) -> Snapshot:
        return self.snapshot


class QuietGate(ThermalGate):
    """ThermalGate whose RAM-pressure check is scripted instead of reading the real machine."""

    demote = False

    def should_demote(self) -> bool:
        return self.demote


def make_config(server: MockLLMServer, tmp: Path) -> Config:
    """Every tier (local and cloud) pointed at one mock server; state under ``tmp``."""
    cfg = Config(home=tmp)
    cfg.state_dir = str(tmp / "state")
    for t in cfg.tiers.values():
        t.base_url = server.v1 if t.protocol == "openai" else server.url
    cfg.validate()
    return cfg


def make_router(server: MockLLMServer, tmp: Path, host: FakeHost = None, **kw) -> Router:
    cfg = kw.pop("config", None) or make_config(server, tmp)
    host = host or FakeHost()
    env = dict(TEST_ENV)
    gate = QuietGate(cfg.thermal, prober=host, env=env)
    return Router(cfg, thermal=gate, env=env, **kw)


def tmpdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="llmrouter_test_")

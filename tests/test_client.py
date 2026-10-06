"""HTTP layer against a mock OpenAI/Anthropic-compatible server."""

import tempfile
import unittest
from pathlib import Path

from helpers import MockLLMServer, TEST_ENV, anthropic_reply, completion
from local_llm_router import client
from local_llm_router.config import Config
from local_llm_router.errors import MissingCredentials, RateLimited, TierBusy, TierError

MSGS = [{"role": "user", "content": "hi"}]


class OpenAIProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = MockLLMServer("  hello  ").start()
        cls.cfg = Config()
        for t in cls.cfg.tiers.values():
            t.base_url = cls.srv.v1 if t.protocol == "openai" else cls.srv.url

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def setUp(self):
        self.srv.requests.clear()
        self.srv.queue.clear()
        self.srv.reply_text = "  hello  "

    def test_success_returns_stripped_text_and_real_usage(self):
        text, usage = client.complete(self.cfg.tier("small"), MSGS, env={})
        self.assertEqual(text, "hello")
        self.assertEqual((usage.tokens_in, usage.tokens_out, usage.real), (11, 7, True))

    def test_request_shape(self):
        client.complete(self.cfg.tier("small"), MSGS, max_tokens=33, temperature=0.0, env={})
        req = self.srv.posts()[0]
        self.assertEqual(req["path"], "/v1/chat/completions")
        self.assertEqual(req["body"]["model"], self.cfg.tier("small").model)
        self.assertEqual(req["body"]["max_tokens"], 33)
        self.assertEqual(req["body"]["messages"], MSGS)
        self.assertNotIn("authorization", req["headers"])    # local tiers send no key

    def test_extra_payload_is_merged(self):
        tier = Config().tier("fast")
        tier.base_url = self.srv.v1
        tier.extra_payload = {"enable_thinking": False}
        client.complete(tier, MSGS, env={})
        self.assertIs(self.srv.posts()[0]["body"]["enable_thinking"], False)

    def test_cloud_key_is_sent_as_bearer(self):
        client.complete(self.cfg.tier("groq"), MSGS, env=TEST_ENV)
        self.assertEqual(self.srv.posts()[0]["headers"]["authorization"], "Bearer test-key")

    def test_missing_cloud_key_raises_before_any_request(self):
        with self.assertRaises(MissingCredentials):
            client.complete(self.cfg.tier("groq"), MSGS, env={})
        self.assertEqual(self.srv.requests, [])

    def test_503_with_busy_contract_is_tierbusy_with_holder(self):
        self.srv.respond(503, {"error": "local-pool-busy", "holder": "model-b", "holder_pid": 4242})
        with self.assertRaises(TierBusy) as ctx:
            client.complete(self.cfg.tier("small"), MSGS, env={})
        self.assertEqual((ctx.exception.holder, ctx.exception.holder_pid), ("model-b", 4242))

    def test_busy_contract_with_status_200_is_still_busy(self):
        self.srv.respond(200, {"error": "local-pool-busy", "holder": "model-b", "holder_pid": 1})
        with self.assertRaises(TierBusy):
            client.complete(self.cfg.tier("small"), MSGS, env={})

    def test_plain_503_is_a_real_fault_not_busy(self):
        self.srv.respond(503, {"error": "overloaded"})
        with self.assertRaises(TierError) as ctx:
            client.complete(self.cfg.tier("small"), MSGS, env={})
        self.assertNotIsInstance(ctx.exception, TierBusy)

    def test_429_and_402_are_rate_limited(self):
        for status in (429, 402):
            with self.subTest(status=status):
                self.srv.respond(status, {"error": "slow down"})
                with self.assertRaises(RateLimited):
                    client.complete(self.cfg.tier("groq"), MSGS, env=TEST_ENV)

    def test_malformed_completion_is_tiererror_not_keyerror(self):
        for bad in ({"choices": []}, {"nope": 1}, {"choices": [{"message": None}]}):
            with self.subTest(bad=bad):
                self.srv.respond(200, bad)
                with self.assertRaises(TierError):
                    client.complete(self.cfg.tier("small"), MSGS, env={})

    def test_non_json_body_is_tiererror(self):
        self.srv.respond(200, b"<html>gateway</html>")
        with self.assertRaises(TierError):
            client.complete(self.cfg.tier("small"), MSGS, env={})

    def test_unreachable_server_is_tiererror(self):
        tier = Config().tier("small")
        tier.base_url = "http://127.0.0.1:1/v1"       # nothing listens on port 1
        with self.assertRaises(TierError):
            client.complete(tier, MSGS, timeout=0.5, env={})

    def test_health_probe(self):
        tier = self.cfg.tier("small")
        self.assertTrue(client.health(tier, env={}))
        self.srv.models_status = 500
        try:
            self.assertFalse(client.health(tier, env={}))
        finally:
            self.srv.models_status = 200
        dead = Config().tier("small")
        dead.base_url = "http://127.0.0.1:1/v1"
        self.assertFalse(client.health(dead, timeout=0.3, env={}))

    def test_cloud_health_is_key_presence_without_a_network_call(self):
        self.assertTrue(client.health(self.cfg.tier("groq"), env=TEST_ENV))
        self.assertFalse(client.health(self.cfg.tier("groq"), env={}))
        self.assertEqual(self.srv.requests, [])


class AnthropicProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = MockLLMServer("claude says hi").start()
        cls.tier = Config().tier("sonnet")
        cls.tier.base_url = cls.srv.url

    @classmethod
    def tearDownClass(cls):
        cls.srv.stop()

    def test_messages_endpoint_system_prompt_and_headers(self):
        msgs = [{"role": "system", "content": "be terse"}, {"role": "user", "content": "hi"}]
        text, usage = client.complete(self.tier, msgs, env=TEST_ENV)
        self.assertEqual(text, "claude says hi")
        self.assertEqual((usage.tokens_in, usage.tokens_out), (5, 3))
        req = self.srv.posts()[-1]
        self.assertEqual(req["path"], "/v1/messages")
        self.assertEqual(req["body"]["system"], "be terse")
        self.assertEqual(req["body"]["messages"], [{"role": "user", "content": "hi"}])
        self.assertEqual(req["headers"]["x-api-key"], "test-key")
        self.assertEqual(req["headers"]["anthropic-version"], client.ANTHROPIC_VERSION)

    def test_multiple_text_blocks_are_joined(self):
        self.srv.respond(200, {"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]})
        self.assertEqual(client.complete(self.tier, MSGS, env=TEST_ENV)[0], "ab")


class Helpers(unittest.TestCase):
    def test_usage_counts_variants(self):
        self.assertEqual(client.usage_counts(completion("x", 4, 2)).tokens_in, 4)
        self.assertEqual(client.usage_counts(anthropic_reply("x")).tokens_out, 3)
        ollama = client.usage_counts({"prompt_eval_count": 9, "eval_count": 6})
        self.assertEqual((ollama.tokens_in, ollama.tokens_out), (9, 6))
        self.assertFalse(client.usage_counts({}).real)
        self.assertFalse(client.usage_counts("junk").real)

    def test_image_part_inlines_a_data_url(self):
        with tempfile.TemporaryDirectory() as d:
            img = Path(d) / "shot.png"
            img.write_bytes(b"\x89PNG-test")
            part = client.image_part(img)
        self.assertTrue(part["image_url"]["url"].startswith("data:image/png;base64,"))

    def test_missing_image_is_rejected(self):
        with self.assertRaises(TierError):
            client.image_part("/nonexistent/nope.png")


if __name__ == "__main__":
    unittest.main(verbosity=2)

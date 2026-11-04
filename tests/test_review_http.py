"""Regression checks for invalid model responses and bounded physical requests."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import tempfile
from unittest.mock import patch

import test_decision_models as fixtures

from odr import (
    CachedAsyncOpenAI,
    EmbedderOpenAI,
    LLMOpenAI,
    ModelDiagnostics,
    SQLiteCache,
)
from odr.models.embeddings import decode_embeddings

NOT_A_NUMBER = float("nan")


class EmbeddingTests(fixtures.DecisionTestCase):
    """Exact input coverage is validated before request cache writes."""

    async def test_invalid_indices_not_cached(self) -> None:
        """A transient duplicate-index response can be replaced by a valid retry."""
        self.queue.responses.extend(
            [
                dict(
                    data=[
                        dict(index=0, embedding=[1, 0]),
                        dict(index=0, embedding=[0, 1]),
                    ]
                ),
                dict(
                    data=[
                        dict(index=1, embedding=[0, 1]),
                        dict(index=0, embedding=[1, 0]),
                    ]
                ),
            ]
        )
        model = EmbedderOpenAI(self.client, "embedding")
        with self.assertRaises(ValueError):
            await model.embed(["first", "second"])
        vectors = await model.embed(["first", "second"])
        self.assertEqual(vectors, [[1, 0], [0, 1]])
        self.assertEqual(len(self.queue.requests), 2)
        self.assertEqual(self.client.stats["actual_inference_calls"], 1)

    def test_strict_embedding_values(self) -> None:
        """Missing indices, strings and booleans never become accepted vectors."""
        invalid = [[True, 0], ["1", 0], [NOT_A_NUMBER, 0], []]
        for vector in invalid:
            with self.subTest(vector=vector), self.assertRaises(ValueError):
                response = dict(data=[dict(index=0, embedding=vector)])
                decode_embeddings(response, count=1)
        with self.assertRaises(ValueError):
            response = dict(data=[dict(index=0, embedding=[1, 0])])
            decode_embeddings(response, count=2)

    async def test_bounded_embedding_batches(self) -> None:
        """Endpoint batches preserve global order and do not send local settings."""
        self.queue.responses.extend(
            [
                dict(
                    data=[
                        dict(index=0, embedding=[1, 0]),
                        dict(index=1, embedding=[0, 1]),
                    ]
                ),
                dict(data=[dict(index=0, embedding=[1, 1])]),
            ]
        )
        model = EmbedderOpenAI(self.client, "embedding", batch_size=2)
        vectors = await model.embed(["a", "b", "c"])
        expected = [[1, 0], [0, 1], [1, 1]]
        self.assertEqual(vectors, expected)
        self.assertEqual(await model.embed([]), [])
        payloads = [json.loads(request.content) for request in self.queue.requests]
        self.assertEqual(
            [payload["input"] for payload in payloads], [["a", "b"], ["c"]]
        )
        self.assertNotIn("batch_size", payloads[0])

    def test_unknown_cache_policy(self) -> None:
        """Misspelled replay cannot silently enable HTTP in either entry point."""
        with self.assertRaises(ValueError):
            CachedAsyncOpenAI("http://localhost/v1", cache_mode="replaay")
        with self.assertRaises(ValueError):
            self.client.cache_mode = "replaay"
        self.assertEqual(str(self.client.cache_mode), "read_write")
        self.assertFalse(self.queue.requests)

    async def test_rendered_budget(self) -> None:
        """Unicode payloads and instructions are measured as actually rendered."""
        self.queue.responses.append(fixtures.completion(json.dumps(self.scores)))
        model = LLMOpenAI(self.client, "chat")
        expected = model.context_size(self.state, self.candidates)
        await model.binary_judge(self.state, self.candidates)
        messages = json.loads(self.queue.requests[0].content)["messages"]
        measured = sum(len(message["content"]) for message in messages)
        self.assertEqual(expected, measured)

    async def test_cache_connection_closed(self) -> None:
        """A completed worker operation closes its connection explicitly."""
        with tempfile.TemporaryDirectory() as folder:
            cache = SQLiteCache(f"{folder}/cache.sqlite3")
            connection = sqlite3.connect(cache.path, check_same_thread=False)
            with patch.object(cache, "_connect", return_value=connection):
                await cache.put("identity", dict(accepted=True))
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")


class DiagnosticTests(fixtures.DecisionTestCase):
    """Per-run counters and latency are isolated from shared global client stats."""

    async def test_singleflight_and_cache_diagnostics(self) -> None:
        """Physical inference, logical waiters and cache hits remain distinguishable."""
        self.queue.responses.append(fixtures.completion("answer"))
        model = LLMOpenAI(self.client, "chat")
        run = ModelDiagnostics()
        with run:
            await asyncio.gather(model.chat_completion([]), model.chat_completion([]))
            await model.chat_completion([])
        self.assertEqual(sum(call.attempts for call in run.calls), 1)
        self.assertEqual(sum(call.shared for call in run.calls), 1)
        self.assertEqual(sum(call.cache_hit for call in run.calls), 1)
        self.assertTrue(
            all(call.accepted and call.elapsed_seconds >= 0 for call in run.calls)
        )
        self.assertEqual(len({call.cache_key for call in run.calls}), 1)

    async def test_concurrent_collectors(self) -> None:
        """Independent tasks collect only their own model operations."""
        self.queue.responses.extend(
            [fixtures.completion("a"), fixtures.completion("b")]
        )
        first, second = await asyncio.gather(
            self.collect("first"), self.collect("second")
        )
        self.assertEqual(len(first.calls), 1)
        self.assertEqual(len(second.calls), 1)
        self.assertNotEqual(first.calls[0].cache_key, second.calls[0].cache_key)

    async def collect(self, model_name: str) -> ModelDiagnostics:
        """Execute a model call in a task-owned collector.

        :param model_name: Distinct fixture model identity.
        :returns: The isolated diagnostics collector.
        """
        run = ModelDiagnostics()
        with run:
            await LLMOpenAI(self.client, model_name).chat_completion([])
        return run

    async def test_invalid_response_usage(self) -> None:
        """Rejected responses retain physical usage without becoming accepted hits."""
        response = dict(data=[], usage=dict(total_tokens=7))
        self.queue.responses.append(response)
        run = ModelDiagnostics()
        with run:
            with self.assertRaises(ValueError):
                await EmbedderOpenAI(self.client, "embedding").embed(["text"])
        self.assertFalse(run.calls[0].accepted)
        self.assertFalse(run.calls[0].cache_hit)
        self.assertEqual(run.calls[0].attempts, 1)
        self.assertEqual(run.calls[0].usage, dict(total_tokens=7))

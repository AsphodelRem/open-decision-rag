"""HTTP contracts, validation and replay for real decision adapters."""

from __future__ import annotations

import asyncio
import json
import unittest

import httpx

from odr import (
    CachedAsyncOpenAI,
    CorpusHierarchy,
    DecisionModel,
    DecisionModelSystemOne,
    DecisionSearchEngine,
    LLMOpenAI,
)
from odr.models.decisions import chat_scores, systemone_scores
from odr.models.types import DecisionState, WireResponse
from odr.schema import Chunk, Document, SearchArguments

SUCCESS_STATUS = 200
INSTALL = "install"
BILLING = "billing"
ANSWERS = "answers"
ANSWER_TYPE = "type"
BINARY = "noul"
SELECTION = "selection"
CHOICE = "choice"
MODEL = "openjev-0.1"
GENERATED_ANSWER = "Install with pip"
TOKEN_BUDGET = 128
TOO_MANY_CHOICES = 256


def completion(text: str) -> WireResponse:
    """Wrap generated content in a wire response.

    :param text: Generated assistant text.
    :returns: Chat response including provider usage.
    """
    return {
        "choices": [{"message": dict(content=text)}],
        "usage": {"total_tokens": 7},
    }


class ResponseQueue:
    """Supply independent wire fixtures and record physical attempts."""

    def __init__(self) -> None:
        """Initialize response and request queues."""
        self.responses: list[WireResponse] = []
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record an attempt and consume a configured response.

        :param request: Request produced by the actual HTTP client.
        :returns: Deterministic successful response.
        """
        self.requests.append(request)
        return httpx.Response(SUCCESS_STATUS, json=self.responses.pop(0))


class DecisionTestCase(unittest.IsolatedAsyncioTestCase):
    """Own test clients and shared candidate context."""

    async def asyncSetUp(self) -> None:
        """Create explicitly owned transports and common inputs."""
        self.candidates = {INSTALL: "Install the package", BILLING: "Monthly invoices"}
        self.scores = {INSTALL: 0.8, BILLING: 0.2}
        self.state: DecisionState = {
            "query": "How do I install?",
            "history": (("user", "I use Linux"),),
            "candidates": self.candidates,
        }
        self.queue = ResponseQueue()
        transport = httpx.AsyncClient(transport=httpx.MockTransport(self.queue))
        self.client = CachedAsyncOpenAI(
            "http://localhost/v1", http_client=transport, api_key="local"
        )
        self.addAsyncCleanup(transport.aclose)
        self.addAsyncCleanup(self.client.__aexit__)

    def binary_response(self) -> WireResponse:
        """Return a native binary fixture.

        :returns: Independent yes probabilities in reversed candidate order.
        """
        return {
            ANSWERS: {
                BILLING: {ANSWER_TYPE: BINARY, BINARY: 0.2},
                INSTALL: {ANSWER_TYPE: BINARY, BINARY: 0.8},
            }
        }

    def choice_response(self) -> WireResponse:
        """Return confidence distinct from the relevance probabilities.

        :returns: A native choice response.
        """
        return {
            ANSWERS: {
                SELECTION: {
                    ANSWER_TYPE: CHOICE,
                    CHOICE: INSTALL,
                    "probabilities": self.scores,
                    "confidence": 0.01,
                }
            }
        }


class SystemOneTests(DecisionTestCase):
    """Native request rendering and caching semantics."""

    async def test_binary_and_replay(self) -> None:
        """Concurrent calls share inference and replay preserves exact scores."""
        self.queue.responses.append(self.binary_response())
        model = DecisionModelSystemOne(self.client, MODEL, samples=1)
        scores = await asyncio.gather(
            *(model.binary_judge(self.state, self.candidates) for index in range(3))
        )
        self.assertTrue(all(score == self.scores for score in scores))
        self.client.cache_mode = "replay"
        self.assertEqual(
            await model.binary_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(len(self.queue.requests), 1)

    async def test_binary_wire(self) -> None:
        """Send all candidates and history together with explicit authorization."""
        self.queue.responses.append(self.binary_response())
        await DecisionModelSystemOne(self.client, MODEL, samples=1).binary_judge(
            self.state, self.candidates
        )
        request = self.queue.requests[0]
        payload = json.loads(request.content)
        self.assertEqual(request.url.path, "/v1/systemone")
        self.assertEqual(request.headers["Authorization"], "Bearer local")
        self.assertEqual(payload["samples"], 1)
        self.assertEqual(payload["state"]["history"], [["user", "I use Linux"]])
        self.assertEqual(set(payload["questions"]), set(self.candidates))

    async def test_choice_and_revision(self) -> None:
        """Use native probabilities and isolate server revisions in the cache."""
        self.queue.responses.extend([self.choice_response(), self.choice_response()])
        first = DecisionModelSystemOne(self.client, MODEL, deployment_revision="v1")
        second = DecisionModelSystemOne(self.client, MODEL, deployment_revision="v2")
        self.assertEqual(
            await first.choice_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(
            await second.choice_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(len(self.queue.requests), 2)
        payload = json.loads(self.queue.requests[0].content)
        self.assertEqual(payload["questions"][SELECTION]["criteria"], self.candidates)

    async def test_empty_and_option_limit(self) -> None:
        """Handle empty inputs locally and reject unsupported choice widths."""
        model = DecisionModelSystemOne(self.client, MODEL)
        self.assertEqual(await model.binary_judge(self.state, {}), {})
        self.assertEqual(await model.choice_judge(self.state, {}), {})
        with self.assertRaises(ValueError):
            await model.choice_judge(
                self.state,
                {str(index): "candidate" for index in range(TOO_MANY_CHOICES)},
            )
        self.assertFalse(self.queue.requests)

    async def test_invalid_not_cached(self) -> None:
        """Retry an invalid native response instead of replaying it."""
        self.queue.responses.extend([{ANSWERS: {}}, self.binary_response()])
        model = DecisionModelSystemOne(self.client, MODEL)
        with self.assertRaises(ValueError):
            await model.binary_judge(self.state, self.candidates)
        self.assertEqual(
            await model.binary_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(len(self.queue.requests), 2)

    async def test_candidate_order_identity(self) -> None:
        """Keep cache entries separate when native question order changes."""
        self.queue.responses.extend([self.binary_response(), self.binary_response()])
        judge = DecisionModelSystemOne(self.client, MODEL).binary_judge
        reordered = dict(reversed(self.candidates.items()))
        await judge(self.state, self.candidates)
        await judge(self.state, reordered)
        self.assertEqual(len(self.queue.requests), 2)


class ChatDecisionTests(DecisionTestCase):
    """Generated estimates and unchanged generation semantics."""

    async def test_judging_and_generation(self) -> None:
        """Binary, choice and ordinary chat have distinct cache identities."""
        self.queue.responses.extend(
            [
                completion(json.dumps(self.scores)),
                completion(json.dumps(self.scores)),
                completion(GENERATED_ANSWER),
            ]
        )
        model = LLMOpenAI(self.client, MODEL, max_tokens=TOKEN_BUDGET)
        self.assertEqual(
            await model.binary_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(
            await model.choice_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(
            await model.chat_completion([{"role": "user", "content": "How?"}]),
            GENERATED_ANSWER,
        )
        self.client.cache_mode = "replay"
        self.assertEqual(
            await model.choice_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(len(self.queue.requests), 3)

    async def test_judge_wire(self) -> None:
        """Request JSON judging while leaving ordinary chat options unchanged."""
        self.queue.responses.extend(
            [completion(json.dumps(self.scores)), completion("answer")]
        )
        model = LLMOpenAI(self.client, MODEL, max_tokens=TOKEN_BUDGET)
        await model.choice_judge(self.state, self.candidates)
        await model.chat_completion([])
        choice, generation = [
            json.loads(request.content) for request in self.queue.requests
        ]
        self.assertEqual(choice["response_format"], {"type": "json_object"})
        self.assertEqual(choice["max_tokens"], TOKEN_BUDGET)
        self.assertIn("sum to one", choice["messages"][0]["content"])
        self.assertEqual(
            json.loads(choice["messages"][1]["content"])["candidates"],
            self.candidates,
        )
        self.assertNotIn("response_format", generation)

    async def test_invalid_not_cached(self) -> None:
        """Failed JSON generation does not poison subsequent identical requests."""
        self.queue.responses.extend(
            [completion("not JSON"), completion(json.dumps(self.scores))]
        )
        model = LLMOpenAI(self.client, MODEL)
        with self.assertRaises(ValueError):
            await model.binary_judge(self.state, self.candidates)
        self.assertEqual(
            await model.binary_judge(self.state, self.candidates), self.scores
        )
        self.assertEqual(await model.binary_judge(self.state, {}), {})
        self.assertEqual(await model.choice_judge(self.state, {}), {})
        self.assertEqual(len(self.queue.requests), 2)

    def test_invalid_probabilities(self) -> None:
        """Reject booleans, strings, nonfinite values and incorrect identities."""
        invalid = [
            {INSTALL: True, BILLING: 0.2},
            {INSTALL: "0.8", BILLING: 0.2},
            {INSTALL: -0.1, BILLING: 1.1},
            {INSTALL: float("nan"), BILLING: 0.2},
            {INSTALL: 0.8},
            {**self.scores, "extra": 0},
            [0.8, 0.2],
        ]
        for scores in invalid:
            with self.subTest(scores=scores), self.assertRaises(ValueError):
                chat_scores(completion(json.dumps(scores)), self.candidates)

    def test_invalid_choice(self) -> None:
        """Reject malformed chat envelopes and unnormalized distributions."""
        response: WireResponse
        for response in (
            {"choices": []},
            completion("not JSON"),
            completion(json.dumps({INSTALL: 0.8, BILLING: 0.8})),
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                chat_scores(response, self.candidates, choice=True)


class NativeValidationTests(DecisionTestCase):
    """Native response discriminators and candidate identities are strict."""

    def test_binary_shapes(self) -> None:
        """Reject malformed envelopes, incorrect types and invalid scores."""
        response: WireResponse
        for response in (
            {},
            {ANSWERS: []},
            {ANSWERS: {INSTALL: {ANSWER_TYPE: BINARY, BINARY: 0.8}}},
            {
                ANSWERS: {
                    identity: {ANSWER_TYPE: CHOICE, BINARY: score}
                    for identity, score in self.scores.items()
                }
            },
            {
                ANSWERS: {
                    identity: {ANSWER_TYPE: BINARY, BINARY: True}
                    for identity in self.candidates
                }
            },
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                systemone_scores(response, self.candidates)

    def test_choice_shapes(self) -> None:
        """Reject wrong question IDs, unknown selections and invalid sums."""
        answer = self.choice_response()[ANSWERS][SELECTION]
        response: WireResponse
        for response in (
            {ANSWERS: {"wrong": answer}},
            {ANSWERS: {SELECTION: {**answer, CHOICE: "unknown"}}},
            {
                ANSWERS: {
                    SELECTION: {**answer, "probabilities": {INSTALL: 0.1, BILLING: 0.1}}
                }
            },
        ):
            with self.subTest(response=response), self.assertRaises(ValueError):
                systemone_scores(response, self.candidates, choice=True)


class RetrievalIntegrationTests(DecisionTestCase):
    """Exercise navigation, choice reranking and answer generation over HTTP."""

    async def test_systemone_retrieval(self) -> None:
        """Native decisions drive the search engine through binary and choice."""
        passage_scores = {
            f"passage:{identity}": score for identity, score in self.scores.items()
        }
        self.queue.responses.extend(
            [
                self.binary_response(),
                {
                    ANSWERS: {
                        identity: {ANSWER_TYPE: BINARY, BINARY: score}
                        for identity, score in passage_scores.items()
                    }
                },
                {
                    ANSWERS: {
                        SELECTION: {
                            ANSWER_TYPE: CHOICE,
                            CHOICE: f"passage:{INSTALL}",
                            "probabilities": passage_scores,
                        }
                    }
                },
                completion(GENERATED_ANSWER),
            ]
        )
        await self.check_retrieval(DecisionModelSystemOne(self.client, MODEL))

    async def test_chat_retrieval(self) -> None:
        """The same LLM instance supplies decision scores and the final answer."""
        passage_scores = {
            f"passage:{identity}": score for identity, score in self.scores.items()
        }
        self.queue.responses.extend(
            [
                completion(json.dumps(self.scores)),
                completion(json.dumps(passage_scores)),
                completion(json.dumps(passage_scores)),
                completion(GENERATED_ANSWER),
            ]
        )
        await self.check_retrieval(LLMOpenAI(self.client, MODEL))

    async def check_retrieval(self, model: DecisionModel) -> None:
        """Build a corpus, query it and replay all calls without HTTP.

        :param model: Production adapter connected to wire fixtures.
        """
        hierarchy = CorpusHierarchy()
        await hierarchy.build_from_docs(
            [
                Document(
                    id=identity,
                    title=text,
                    chunks=(Chunk(id=identity, doc_id=identity, text=text),),
                )
                for identity, text in self.candidates.items()
            ]
        )
        engine = DecisionSearchEngine(
            hierarchy,
            model,
            llm=LLMOpenAI(self.client, MODEL),
            search_settings=SearchArguments(return_top_k=2, choice_top_m=2),
        )
        response = await engine.query(self.state["query"])
        self.assertEqual(response.response, GENERATED_ANSWER)
        self.assertEqual(response.retrieval.passages[0].document_id, INSTALL)
        self.assertEqual(
            response.retrieval.passages[0].choice_score, self.scores[INSTALL]
        )
        self.client.cache_mode = "replay"
        self.assertEqual(await engine.query(self.state["query"]), response)
        self.assertEqual(len(self.queue.requests), 4)

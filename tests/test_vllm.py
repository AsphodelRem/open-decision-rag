"""Synthetic exact-token contracts for direct vLLM diffusion reads."""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from unittest.mock import patch

import test_decision_models as fixtures

from odr import (
    CachedAsyncOpenAI,
    CorpusHierarchy,
    DecisionModelVLLM,
    DecisionSearchEngine,
    SQLiteCache,
    VLLMDecisionArguments,
)
from odr import schema as records
from odr.models import vllm_readout, vllm_types
from odr.models.types import Messages, WireResponse
from odr.models.vllm_canvas import THOUGHT_SCAFFOLD, CanvasCompiler

MODEL = "dgemma"
VOCABULARY_SIZE = 1000
ANSWER_POSITION = 5
YES_ID = 100
NO_ID = 101
LABEL_OFFSET = 200
CHOICE_OVERFLOW = 53
YES_LABEL = "yes"
NO_LABEL = "no"
POSITIVE_INFINITY = float("inf")
NOT_A_NUMBER = float("nan")
CANVAS_WIDTH = 16
LOGPROB_OFFSET = 10000
EXPECTED_TEMPLATE = (8, 10, 9, 11, 12, YES_ID, 7)
EXPECTED_POSITIVE = 1 / (1 + math.exp(-1))


def configuration(revision: str = "deployment-v1") -> VLLMDecisionArguments:
    """Return explicit identities for a synthetic tokenizer and server.

    :param revision: Deployment version used by request and plan caches.
    :returns: Validated test configuration without real model identities.
    """
    return VLLMDecisionArguments(
        tokenizer_name="fixture-tokenizer",
        tokenizer_revision="fixture-v1",
        deployment_revision=revision,
    )


class FixtureTokenizer:
    """Synthetic tokenization; its IDs are not real DiffusionGemma token IDs."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Represent allowed labels at a known single shared template slot.

        :param text: Special marker or one-question answer template.
        :param add_special_tokens: Unused fixture compatibility parameter.
        :returns: Deterministic synthetic token IDs.
        """
        special = {"<pad>": 0, "<turn|>": 7, "<|channel>": 8, "<channel|>": 9}
        identity = special.get(text)
        if identity is not None:
            return [identity]
        if text == THOUGHT_SCAFFOLD:
            return [8, 10, 9]
        label = text.removeprefix("q1: ")
        identity = {YES_LABEL: YES_ID, NO_LABEL: NO_ID}.get(label)
        if identity is None:
            identity = ord(label) + LABEL_OFFSET
        return [11, 12, identity]

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Decode synthetic special markers for roundtrip compatibility checks.

        :param token_ids: Single synthetic boundary token.
        :param skip_special_tokens: Unused fixture compatibility parameter.
        :returns: Original boundary spelling.
        """
        return {0: "<pad>", 7: "<turn|>", 8: "<|channel>", 9: "<channel|>"}[
            token_ids[0]
        ]

    def apply_chat_template(self, messages: Messages, **settings: object) -> list[int]:
        """Return synthetic template tokens including fixed boundary overhead.

        :param messages: Fully rendered provider messages.
        :param settings: Unused fixture template options.
        :returns: A synthetic prompt token sequence.
        """
        return [1 for message in messages for character in message["content"]]

    def __len__(self) -> int:
        """Return a synthetic vocabulary bound.

        :returns: Exclusive upper bound for fixture token IDs.
        """
        return VOCABULARY_SIZE


class IncompatibleTokenizer(FixtureTokenizer):
    """Change a label's width while retaining valid special-token handling."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Deliberately split the negative label into two tokens.

        :param text: Template or boundary to tokenize.
        :param add_special_tokens: Unused fixture compatibility parameter.
        :returns: An incompatible negative template or valid base tokenization.
        """
        tokens = super().encode(text, add_special_tokens=add_special_tokens)
        return [*tokens, 1] if text == "q1: no" else tokens


def plan(labels: tuple[str, ...] = (YES_LABEL, NO_LABEL)) -> vllm_types.CanvasPlan:
    """Compile an independently specified synthetic tokenizer.

    :param labels: Ordered allowed answer labels.
    :returns: Compiler output with verified label positions.
    """
    return CanvasCompiler(configuration(), FixtureTokenizer()).compile(labels)


def wire_response(tokens: list[int] | None = None) -> WireResponse:
    """Construct raw vLLM response data without using adapter readout helpers.

    :param tokens: Synthetic token identities in response order.
    :returns: Raw token evidence with unrelated dominant alternatives.
    """
    identities = [YES_ID, NO_ID] if tokens is None else tokens
    rows: list[WireResponse] = [
        {"top_logprobs": []} for index in range(ANSWER_POSITION)
    ]
    rows.append(
        {
            "top_logprobs": [
                {"token": "token_id:999", "logprob": -0.01},
                *(
                    {
                        "token": f"token_id:{identity}",
                        "logprob": -LOGPROB_OFFSET - index,
                    }
                    for index, identity in enumerate(identities)
                ),
            ]
        }
    )
    return {
        "choices": [
            {
                "logprobs": dict(content=rows),
                "message": dict(content="ignored generated text"),
            }
        ],
        "usage": {"prompt_tokens": 42},
    }


class CanvasTests(unittest.TestCase):
    """Template positions and tokenizer compatibility are checked before HTTP."""

    def test_template(self) -> None:
        """A label changes one token after the thought scaffold and question ID."""
        compiled = plan()
        self.assertEqual(compiled.position, ANSWER_POSITION)
        self.assertEqual(compiled.label_ids, [YES_ID, NO_ID])
        expected = list(EXPECTED_TEMPLATE)
        self.assertEqual(compiled.canvas[: len(expected)], expected)
        self.assertEqual(len(compiled.canvas), CANVAS_WIDTH)

    def test_incompatible_tokenizer(self) -> None:
        """Refuse labels whose contextual tokenizations have different widths."""
        with self.assertRaises(ValueError):
            CanvasCompiler(configuration(), IncompatibleTokenizer()).compile(
                (YES_LABEL, NO_LABEL)
            )

    def test_bad_token_identity(self) -> None:
        """Reject cached template IDs outside the recorded tokenizer vocabulary."""
        with self.assertRaises(ValueError):
            vllm_types.CanvasPlan(
                canvas=[0],
                position=0,
                output_length=1,
                label_ids=[YES_ID, NO_ID],
                labels=[YES_LABEL, NO_LABEL],
                vocabulary_size=1,
            )

    def test_readout(self) -> None:
        """Stable softmax uses exact IDs even for very unlikely answer labels."""
        scores = vllm_readout.read_probabilities(wire_response(), plan())
        self.assertAlmostEqual(scores[0], EXPECTED_POSITIVE)
        self.assertAlmostEqual(sum(scores), 1)

    def test_missing_and_duplicate_labels(self) -> None:
        """No missing-token floor or duplicate-token replacement is allowed."""
        for tokens in ([YES_ID], [YES_ID, NO_ID, YES_ID]):
            with self.subTest(tokens=tokens), self.assertRaises(ValueError):
                vllm_readout.read_probabilities(wire_response(tokens), plan())

    def test_invalid_logprobs(self) -> None:
        """Refuse nonnumeric, positive, nonfinite and text-only evidence."""
        invalid = [True, "-1", 1, NOT_A_NUMBER, POSITIVE_INFINITY, -POSITIVE_INFINITY]
        for logprob in invalid:
            response = wire_response()
            rows = response["choices"][0]["logprobs"]["content"]
            entry = rows[ANSWER_POSITION]["top_logprobs"][1]
            entry["logprob"] = logprob
            with self.subTest(logprob=logprob), self.assertRaises(ValueError):
                vllm_readout.read_probabilities(response, plan())


class VLLMTests(fixtures.DecisionTestCase):
    """Physical HTTP reads, shared context and persistent offline replay."""

    async def test_binary_wire(self) -> None:
        """Independent reads keep the full candidate context and exact IDs."""
        self.queue.responses.extend([wire_response(), wire_response()])
        model = DecisionModelVLLM(
            self.client, MODEL, configuration(), FixtureTokenizer()
        )
        scores = await model.binary_judge(self.state, self.candidates)
        self.assertAlmostEqual(scores[fixtures.INSTALL], EXPECTED_POSITIVE)
        self.assertEqual(len(self.queue.requests), 2)
        payload = json.loads(self.queue.requests[0].content)
        request = self.queue.requests[0]
        self.assertEqual(request.url.path, "/v1/chat/completions")
        self.assertEqual(payload["logprob_token_ids"], [YES_ID, NO_ID])
        self.assertTrue(payload["return_tokens_as_token_ids"])
        self.assertTrue(payload["vllm_xargs"]["diffusion_read_only"])
        self.assertEqual(
            json.loads(payload["messages"][1]["content"])["candidates"],
            self.candidates,
        )

    async def test_choice(self) -> None:
        """Choice IDs map back from ordered labels rather than generated text."""
        self.queue.responses.append(
            wire_response([ord("A") + LABEL_OFFSET, ord("B") + LABEL_OFFSET])
        )
        model = DecisionModelVLLM(
            self.client, MODEL, configuration(), FixtureTokenizer()
        )
        scores = await model.choice_judge(self.state, self.candidates)
        self.assertAlmostEqual(sum(scores.values()), 1)
        self.assertGreater(scores[fixtures.INSTALL], scores[fixtures.BILLING])

    async def test_persistent_replay(self) -> None:
        """A fresh adapter replays both plans and responses without a tokenizer."""
        with tempfile.TemporaryDirectory() as folder:
            self.client.cache = SQLiteCache(f"{folder}/requests.sqlite3")
            self.queue.responses.extend([wire_response(), wire_response()])
            model = DecisionModelVLLM(
                self.client, MODEL, configuration(), FixtureTokenizer()
            )
            scores = await model.binary_judge(self.state, self.candidates)
            async with CachedAsyncOpenAI(
                "http://localhost/v1",
                cache=SQLiteCache(f"{folder}/requests.sqlite3"),
                cache_mode="replay",
            ) as replay:
                with patch(
                    "odr.models.vllm_canvas.import_module",
                    side_effect=AssertionError("Tokenizer loading is forbidden"),
                ):
                    self.assertEqual(
                        await DecisionModelVLLM(
                            replay, MODEL, configuration()
                        ).binary_judge(self.state, self.candidates),
                        scores,
                    )
                self.assertEqual(replay.stats["network_attempts"], 0)

    async def test_invalid_not_cached(self) -> None:
        """A missing exact logprob is rejected before the wire cache is written."""
        self.queue.responses.extend([wire_response([YES_ID]), wire_response()])
        model = DecisionModelVLLM(
            self.client, MODEL, configuration(), FixtureTokenizer()
        )
        with self.assertRaises(ValueError):
            await model.binary_judge(self.state, {fixtures.INSTALL: "install"})
        self.assertTrue(
            await model.binary_judge(self.state, {fixtures.INSTALL: "install"})
        )
        self.assertEqual(len(self.queue.requests), 2)

    async def test_empty_singleton_and_replay_miss(self) -> None:
        """No network or tokenizer loading is needed for forced decisions."""
        model = DecisionModelVLLM(self.client, MODEL, configuration())
        self.assertEqual(await model.binary_judge(self.state, {}), {})
        self.assertEqual(await model.choice_judge(self.state, {}), {})
        self.assertEqual(
            await model.choice_judge(self.state, {fixtures.INSTALL: "install"}),
            {fixtures.INSTALL: 1},
        )
        self.client.cache_mode = "replay"
        with self.assertRaises(LookupError):
            await model.binary_judge(self.state, self.candidates)
        self.assertFalse(self.queue.requests)

    async def test_choice_limit(self) -> None:
        """Refuse unsupported label counts before loading a tokenizer."""
        model = DecisionModelVLLM(self.client, MODEL, configuration())
        with self.assertRaises(ValueError):
            await model.choice_judge(
                self.state,
                {str(index): "candidate" for index in range(CHOICE_OVERFLOW)},
            )
        self.assertFalse(self.queue.requests)

    async def test_navigation_and_choice(self) -> None:
        """Direct native probabilities drive document and passage navigation."""
        hierarchy = CorpusHierarchy()
        await hierarchy.build_from_docs(
            [
                records.Document(
                    id=identity,
                    title=text,
                    chunks=(records.Chunk(id=identity, doc_id=identity, text=text),),
                )
                for identity, text in self.candidates.items()
            ]
        )
        self.queue.responses.extend(
            [
                wire_response(),
                wire_response(),
                wire_response(),
                wire_response(),
                wire_response([ord("A") + LABEL_OFFSET, ord("B") + LABEL_OFFSET]),
            ]
        )
        engine = DecisionSearchEngine(
            hierarchy,
            DecisionModelVLLM(self.client, MODEL, configuration(), FixtureTokenizer()),
            search_settings=records.SearchArguments(return_top_k=2, choice_top_m=2),
        )
        retrieval = await engine.search(self.state["query"])
        self.assertEqual(retrieval.passages[0].document_id, fixtures.INSTALL)
        self.assertIsNotNone(retrieval.passages[0].choice_score)
        self.client.cache_mode = "replay"
        self.assertEqual(await engine.search(self.state["query"]), retrieval)
        self.assertEqual(len(self.queue.requests), 5)

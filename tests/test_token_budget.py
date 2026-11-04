"""Exact renderer token budgets and tokenizer-free budget replay."""

from __future__ import annotations

import json
from unittest.mock import patch

import test_decision_models as fixtures
import test_vllm as native
from test_review_data import documents

from odr import (
    CorpusHierarchy,
    DecisionModelVLLM,
    DecisionSearchEngine,
    MockDecisionModel,
)
from odr.models.types import WireResponse
from odr.schema import SearchArguments

TOKEN_BUDGET = 3000
PASSAGE_LENGTH = 1000


class TokenBudgetTests(fixtures.DecisionTestCase):
    """Synthetic tokenization verifies complete prompt-plus-output accounting."""

    async def test_complete_prompt_and_replay(self) -> None:
        """Prompt counts match dispatch and replay from cache without a tokenizer."""
        model = DecisionModelVLLM(
            self.client, native.MODEL, native.configuration(), native.FixtureTokenizer()
        )
        count = await model.context_tokens(self.state, self.candidates)
        self.queue.responses.extend([native.wire_response(), native.wire_response()])
        await model.binary_judge(self.state, self.candidates)
        payloads = [json.loads(request.content) for request in self.queue.requests]
        expected = max(
            sum(len(message["content"]) for message in payload["messages"])
            + payload["max_tokens"]
            for payload in payloads
        )
        self.assertEqual(count, expected)
        self.client.cache_mode = "replay"
        with patch(
            "odr.models.vllm_canvas.import_module",
            side_effect=AssertionError("Tokenizer must not load"),
        ):
            replay = DecisionModelVLLM(
                self.client, native.MODEL, native.configuration()
            )
            self.assertEqual(
                await replay.context_tokens(self.state, self.candidates), count
            )

    async def test_split_before_inference(self) -> None:
        """Passage batches split according to rendered prompts before HTTP dispatch."""
        source = documents(2)[0]
        source = source.model_copy(
            update=dict(
                chunks=tuple(
                    chunk.model_copy(update=dict(text="я" * PASSAGE_LENGTH))
                    for chunk in source.chunks
                )
            )
        )
        corpus = await CorpusHierarchy().build_from_docs([source])
        self.queue.responses.extend(
            [native.wire_response(), native.wire_response(), native.wire_response()]
        )
        model = DecisionModelVLLM(
            self.client, native.MODEL, native.configuration(), native.FixtureTokenizer()
        )
        engine = DecisionSearchEngine(
            corpus,
            model,
            search_settings=SearchArguments(max_context_tokens=TOKEN_BUDGET),
        )
        await engine.search("query")
        payloads = [json.loads(request.content) for request in self.queue.requests]
        self.assertEqual(len(payloads), 3)
        self.assert_budget(payloads)

    def assert_budget(self, payloads: list[WireResponse]) -> None:
        """Check physical prompt sizes and the split passage candidate contexts.

        :param payloads: Captured raw vLLM requests.
        """
        for payload in payloads:
            rendered = sum(len(message["content"]) for message in payload["messages"])
            self.assertLessEqual(rendered + payload["max_tokens"], TOKEN_BUDGET)
        for query_payload in payloads[-2:]:
            state = json.loads(query_payload["messages"][1]["content"])
            self.assertEqual(len(state["candidates"]), 1)

    async def test_unsupported_token_counter(self) -> None:
        """Requesting exact tokens from an unsupported provider fails explicitly."""
        corpus = await CorpusHierarchy().build_from_docs(documents())
        engine = DecisionSearchEngine(
            corpus,
            MockDecisionModel(),
            search_settings=SearchArguments(max_context_tokens=TOKEN_BUDGET),
        )
        with self.assertRaisesRegex(ValueError, "cannot enforce"):
            await engine.search("query")

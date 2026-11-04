"""Regression checks for conversational rendering, source IDs and choice metadata."""

from __future__ import annotations

import json
import unittest

from test_review_data import documents

from odr import CorpusHierarchy, DecisionSearchEngine, MockDecisionModel, MockLLM
from odr.evaluation.metrics import evaluate
from odr.evaluation.types import EvaluationRecord
from odr.models.types import Candidates, DecisionState, Messages, Scores
from odr.schema import SearchArguments, SearchQuery

QUERY = "install"
FIRST_CHUNK = "p0"


class RecordingJudge(MockDecisionModel):
    """Independent binary and choice rankings with recorded model contexts."""

    def __init__(self) -> None:
        """Configure opposing rankings to reveal source-order and promotion bugs."""
        self.states: list[DecisionState] = []
        self.options: Candidates = {}
        self.binary = dict(
            zip((f"passage:p{index}" for index in range(3)), (0.1, 0.9, 0.7))
        )
        self.choice = dict(
            zip((f"passage:p{index}" for index in range(3)), (0.6, 0.3, 0.1))
        )

    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Record full query history and return a controlled binary ranking.

        :param state: Shared query context.
        :param candidates: Current candidate mapping.
        :returns: Independent binary probabilities.
        """
        self.states.append(state)
        return {identity: self.binary.get(identity, 1) for identity in candidates}

    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Record shortlist order and return source-keyed choice probabilities.

        :param state: Shared query context.
        :param candidates: Choice shortlist mapping.
        :returns: Normalized choice probabilities.
        """
        self.options = candidates
        return {identity: self.choice[identity] for identity in candidates}


class RecordingGenerator(MockLLM):
    """Record final answer messages independently of relevance decisions."""

    def __init__(self) -> None:
        """Initialize the message log."""
        self.messages: Messages = []

    async def chat_completion(self, messages: Messages) -> str:
        """Capture answer instructions, history and sources.

        :param messages: Actual answer generation input.
        :returns: Deterministic answer text.
        """
        self.messages = messages
        return "Answer"


class SearchContractTests(unittest.IsolatedAsyncioTestCase):
    """Source rendering and evaluation contracts with controlled decisions."""

    async def asyncSetUp(self) -> None:
        """Build a prepared corpus with descriptive title and heading paths."""
        self.corpus = CorpusHierarchy()
        source = documents(3)[0]
        source = source.model_copy(
            update=dict(
                title="Installation manual",
                chunks=tuple(
                    chunk.model_copy(update=dict(heading_path=("Setup",)))
                    for chunk in source.chunks
                ),
            )
        )
        await self.corpus.build_from_docs([source])
        self.judge = RecordingJudge()
        self.generator = RecordingGenerator()
        self.engine = DecisionSearchEngine(
            self.corpus,
            self.judge,
            llm=self.generator,
            search_settings=SearchArguments(
                return_top_k=3, choice_top_m=3, choice_mode="promote_top1"
            ),
        )

    async def test_generation_history(self) -> None:
        """Final generation retains conversational context and explicit sources."""
        query = SearchQuery(
            text="And when?", messages=(("user", "When was it released?"),)
        )
        await self.engine.query(query)
        self.assertEqual(self.judge.states[0]["history"], query.messages)
        self.assertIn(
            dict(role="user", content="When was it released?"), self.generator.messages
        )
        context = json.loads(self.generator.messages[-1]["content"])
        self.assertEqual(context["query"], query.text)
        self.assertEqual(context["sources"][0]["source_id"], FIRST_CHUNK)

    async def test_source_order_and_choice_scores(self) -> None:
        """Choice uses reading order and preserves every evaluated option score."""
        retrieval = await self.engine.search(QUERY)
        self.assertEqual(list(self.judge.options), list(self.judge.binary))
        self.assertEqual(
            [passage.chunk_id for passage in retrieval.passages],
            [FIRST_CHUNK, "p1", "p2"],
        )
        self.assertTrue(
            all(passage.choice_score is not None for passage in retrieval.passages)
        )
        for rendered in self.judge.options.values():
            self.assertIn("Installation manual", rendered)
            self.assertIn("Setup", rendered)

    async def test_source_and_node_qrels(self) -> None:
        """Default source chunk IDs and explicit legacy node IDs both evaluate."""
        source_record: EvaluationRecord = dict(
            id="source", query=QUERY, qrels={FIRST_CHUNK: 1}
        )
        source = await evaluate(self.engine, [source_record], cutoff=3)
        nodes = await evaluate(
            self.engine,
            [
                dict(
                    id="nodes",
                    query=QUERY,
                    qrels={"passage:p0": 1},
                    id_space="node",
                )
            ],
            cutoff=3,
        )
        self.assertEqual(source[0]["hit"], 1)
        self.assertEqual(nodes[0]["hit"], 1)
        self.assertEqual(source[0]["cutoff"], 3)
        self.assertEqual(source[0]["returned_count"], 3)

    async def test_cutoff_validation(self) -> None:
        """Misleading evaluation depth fails before relevance requests."""
        with self.assertRaises(ValueError):
            await evaluate(
                self.engine, [dict(id="query", query=QUERY, qrels={FIRST_CHUNK: 1})]
            )
        self.assertFalse(self.judge.states)

    async def test_structured_trace(self) -> None:
        """Numerical traces contain scoring and branch/choice selection events."""
        retrieval = await self.engine.search(QUERY)
        events = [json.loads(event) for event in retrieval.trace]
        self.assertEqual(
            {event["stage"] for event in events}, {"binary", "branches", "choice"}
        )
        self.assertTrue(
            all("scores" in event and "selected" in event for event in events)
        )

    async def test_budget_rejection(self) -> None:
        """Oversized candidates fail without emitting an empty scoring request."""
        engine = DecisionSearchEngine(
            self.corpus,
            self.judge,
            search_settings=SearchArguments(max_context_chars=1),
        )
        with self.assertRaises(ValueError):
            await engine.search(QUERY)
        self.assertFalse(self.judge.states)


class EvaluationBoundaryTests(unittest.IsolatedAsyncioTestCase):
    """Gold labels are checked against the explicitly selected corpus namespace."""

    async def test_wrong_namespace(self) -> None:
        """Internal node labels cannot silently produce false source-ID misses."""
        corpus = await CorpusHierarchy().build_from_docs(documents())
        judge = RecordingJudge()
        engine = DecisionSearchEngine(corpus, judge)
        record: EvaluationRecord = dict(
            id="wrong", query=QUERY, qrels={"passage:p0": 1}
        )
        with self.assertRaises(ValueError):
            await evaluate(engine, [record], cutoff=1)
        self.assertFalse(judge.states)

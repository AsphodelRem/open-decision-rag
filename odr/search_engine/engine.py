"""Hierarchical shared-context binary navigation and choice reranking."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.corpus import facade, index
    from odr.models import base, types
    from odr.schema.records import HierarchyNode
import asyncio
import math

from odr.schema import (
    DecisionSearchRetrieve,
    RetrievedPassage,
    SearchArguments,
    SearchEngineResponse,
    SearchQuery,
    fingerprint,
)
from odr.schema.provenance import component_identity
from odr.search_engine import rendering, sources
from odr.search_engine.scoring import CandidateScoring

PROBABILITY_TOLERANCE = 1e-6


def validate_scores(
    scores: types.Scores, candidates: types.Candidates, choice: bool = False
) -> None:
    """Require exact candidate coverage and finite probabilities.

    :param scores: Candidate identities mapped to probabilities.
    :param candidates: Candidate identities mapped to rendered descriptions.
    :param choice: Whether probabilities must form a normalized distribution.
    :raises ValueError: If the input, stored data, or operation result is invalid.
    """
    if set(scores) != set(candidates):
        raise ValueError("Decision candidate IDs do not match")
    finite = all((math.isfinite(score) for score in scores.values()))
    valid = [0 <= score <= 1 for score in scores.values()]
    if not finite or not all(valid):
        raise ValueError("Invalid decision probability")
    if choice and (
        not math.isclose(sum(scores.values()), 1, abs_tol=PROBABILITY_TOLERANCE)
    ):
        raise ValueError("Choice distribution must sum to one")


class DecisionSearchEngine:
    """Retrieval over a public index; no model work during index reads."""

    def __init__(
        self,
        hierarchy: facade.CorpusHierarchy | index.HierarchyIndex,
        decision_model: base.DecisionModel,
        llm: base.LLM | None = None,
        search_settings: SearchArguments | None = None,
    ) -> None:
        """Configure retrieval and optional answer generation.

        :param hierarchy: Built corpus facade or read-only hierarchy index.
        :param decision_model: Injected shared-context relevance and choice judge.
        :param llm: Injected generation model.
        :param search_settings: Validated navigation and ranking settings.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        bound_index = hierarchy.index if hasattr(hierarchy, "index") else hierarchy
        self.index = cast("index.HierarchyIndex", bound_index)
        self.decision_model = decision_model
        self.llm = llm
        self.settings = search_settings or SearchArguments()
        if self.settings.profile != "generic":
            raise ValueError(
                "This implementation supports only the generic navigation profile"
            )

    async def search(
        self, query: str | SearchQuery, session: object | None = None
    ) -> DecisionSearchRetrieve:
        """Navigate selected branches, then rank all reached passages.

        :param query: User query text or a validated query with conversation history.
        :param session: Reserved session argument; nonempty sessions are unsupported.
        :returns: Ranked passages, corpus identity and numerical decision trace.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        if session is not None:
            raise ValueError(
                "Session navigation is unavailable; pass explicit query messages"
            )
        query = SearchQuery(text=query) if isinstance(query, str) else query
        traversal = _Traversal(self, query)
        await traversal.navigate()
        passages = await traversal.rank()
        return DecisionSearchRetrieve(
            query=query,
            passages=passages,
            corpus_revision=self.index.revision,
            config_fingerprint=fingerprint(
                dict(
                    settings=self.settings.model_dump(),
                    judge=component_identity(self.decision_model),
                )
            ),
            trace=tuple(traversal.trace),
        )

    async def query(
        self, query: str | SearchQuery, session: object | None = None
    ) -> SearchEngineResponse:
        """Generate an answer only through an explicitly supplied base.LLM.

        :param query: User query text or a validated query with conversation history.
        :param session: Reserved session argument; nonempty sessions are unsupported.
        :returns: Generated answer together with its original retrieval.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        if self.llm is None:
            raise ValueError("query requires an base.LLM; use search for retrieval")
        retrieval = await self.search(query, session=session)
        answer = await self.llm.chat_completion(rendering.answer_messages(retrieval))
        return SearchEngineResponse(
            query=retrieval.query, response=answer, retrieval=retrieval
        )

    def search_sync(
        self, query: str | SearchQuery, session: object | None = None
    ) -> DecisionSearchRetrieve:
        """Run retrieval outside an active event loop.

        :param query: User query text or a validated query with conversation history.
        :param session: Reserved session argument; nonempty sessions are unsupported.
        :returns: Ranked passages, corpus identity and numerical decision trace.
        """
        return asyncio.run(self.search(query, session=session))

    def query_sync(
        self, query: str | SearchQuery, session: object | None = None
    ) -> SearchEngineResponse:
        """Run retrieval and generation outside an active event loop.

        :param query: User query text or a validated query with conversation history.
        :param session: Reserved session argument; nonempty sessions are unsupported.
        :returns: Generated answer together with its original retrieval.
        """
        return asyncio.run(self.query(query, session=session))


class _Traversal(CandidateScoring):
    """Query-local navigation and passage ranking state."""

    def __init__(self, engine: DecisionSearchEngine, query: SearchQuery) -> None:
        """Initialize query-local traversal, passage and score state.

        :param engine: Configured retrieval engine.
        :param query: User query text or a validated query with conversation history.
        """
        self.engine = engine
        self.query = query
        self.trace: list[str] = []
        self.terminals: list[HierarchyNode] = []
        self.sources = sources.ResolvedSources(engine.index)

    async def navigate(self) -> None:
        """Expand selected internal branches until all reached nodes are terminal.

        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        frontier = await self.engine.index.children(self.engine.index.root_id)
        depth = 0
        self.expanded = 0
        while frontier:
            if depth >= self.engine.settings.max_depth:
                raise ValueError("Search depth budget exceeded")
            self.terminals.extend((node for node in frontier if node.kind == "passage"))
            internal = [node for node in frontier if node.kind != "passage"]
            selected = await self._select(internal)
            children = await asyncio.gather(
                *(self.engine.index.children(node.id) for node in selected)
            )
            frontier = [node for batch in children for node in batch]
            depth += 1

    async def score(self, nodes: Sequence[HierarchyNode]) -> types.Scores:
        """Merge shared-context binary scores from deterministic batches.

        :param nodes: Hierarchy nodes or their identity mapping.
        :returns: Independent binary probabilities for the candidate nodes.
        """
        if not nodes:
            return {}
        batches = await self._batches(await self._candidates(nodes))
        responses = await asyncio.gather(*(self._judge(batch) for batch in batches))
        for batch, scores in zip(batches, responses):
            self.trace.append(
                rendering.trace_event(
                    "binary", scores, tuple(batch), self.context_size(batch)
                )
            )
        return dict(pair for response in responses for pair in response.items())

    async def rank(self) -> tuple[RetrievedPassage, ...]:
        """Rank terminal passages and apply the configured choice policy.

        :returns: Final retrieved passages with one-based ranks.
        """
        scores = await self.score(self.terminals)
        records = await asyncio.gather(
            *(self._record(node, scores[node.id]) for node in self.terminals)
        )
        records.sort(key=lambda record: -record.ranking_score)
        pool = self.engine.settings.candidate_pool_size
        records = records if pool is None else records[:pool]
        records = await self._choice(records)
        return tuple(
            (
                record.model_copy(update={"rank": rank + 1})
                for (rank, record) in enumerate(
                    records[: self.engine.settings.return_top_k]
                )
            )
        )

    async def _select(self, internal: Sequence[HierarchyNode]) -> list[HierarchyNode]:
        """Retain globally ranked branches within the expansion budget.

        :param internal: Internal hierarchy candidates available for expansion.
        :returns: Internal candidates retained for expansion.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        scores = await self.score(internal)
        selected = sorted(
            internal, key=lambda node: (-scores[node.id], node.position, node.id)
        )
        selected = selected[: self.engine.settings.branch_top_k]
        self.expanded += len(selected)
        if self.expanded > self.engine.settings.max_expanded_nodes:
            raise ValueError("Expanded node budget exceeded")
        self.sources.document_scores.update(
            {node.id: scores[node.id] for node in internal if node.kind == "document"}
        )
        self.trace.append(
            rendering.trace_event(
                "branches", scores, tuple(node.id for node in selected)
            )
        )
        return selected

    async def _record(self, node: HierarchyNode, score: float) -> RetrievedPassage:
        """Resolve a passage source and compute its configured ranking score.

        :param node: Hierarchy node to inspect.
        :param score: Independent binary relevance probability.
        :returns: A retrieved passage with its source path and ranking scores.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        return self.sources.retrieved(node, score, self.engine.settings.scoring)

    async def _choice(self, records: list[RetrievedPassage]) -> list[RetrievedPassage]:
        """Reorder the retained shortlist or promote only its winning passage.

        :param records: Ordered records to read, rank, validate, or evaluate.
        :returns: Records after applying the configured shortlist ordering policy.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        count = min(self.engine.settings.choice_top_m, len(records))
        if count < 2:
            return records
        candidates = rendering.choice_candidates(records[:count], self.sources)
        if not await self._fits(candidates, choice=True):
            raise ValueError("Choice context limit exceeded")
        scores = await self.engine.decision_model.choice_judge(
            self._state(candidates), candidates
        )
        validate_scores(scores, candidates, choice=True)
        ranked = [
            record.model_copy(update={"choice_score": scores[record.node_id]})
            for record in records[:count]
        ]
        ranked.sort(key=lambda record: -scores[record.node_id])
        self.trace.append(
            rendering.choice_trace(
                scores,
                ranked,
                self.engine.settings.choice_mode == "promote_top1",
                self.context_size(candidates, choice=True),
            )
        )
        if self.engine.settings.choice_mode == "promote_top1":
            return rendering.promote_winner(ranked, records)
        return [*ranked, *records[count:]]

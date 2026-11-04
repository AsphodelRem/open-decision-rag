"""Internal CandidateScoring operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

from odr.search_engine.sources import ResolvedSources

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.models.types import Candidates, DecisionState, Scores
    from odr.schema.records import HierarchyNode
    from odr.schema.responses import SearchQuery
    from odr.search_engine.engine import DecisionSearchEngine
PROBABILITY_TOLERANCE = 1e-6


class CandidateScoring:
    """Shared candidate rendering, deterministic batching and judging.

    :ivar engine: Configured retrieval engine.
    :ivar query: Original validated query.
    :ivar trace: Numerical traversal and candidate batch events.
    """

    engine: DecisionSearchEngine
    query: SearchQuery
    trace: list[str]
    sources: ResolvedSources

    def context_size(self, candidates: Candidates, choice: bool = False) -> int:
        """Measure provider-rendered message content before dispatch.

        :param candidates: Candidate identities and rendered source descriptions.
        :param choice: Whether to estimate a choice request.
        :returns: Rendered character estimate including instructions and options.
        """
        return self.engine.decision_model.context_size(
            self._state(candidates), candidates, choice
        )

    async def _candidates(self, nodes: Sequence[HierarchyNode]) -> Candidates:
        """Resolve internal descriptions and terminal passage texts.

        :param nodes: Hierarchy nodes or their identity mapping.
        :returns: Candidate IDs mapped to rendered descriptions or passage texts.
        """
        await self.sources.resolve(nodes)
        return {
            node.id: self.sources.render(node)
            for node in sorted(nodes, key=self.sources.order)
        }

    async def _batches(self, candidates: Candidates) -> list[Candidates]:
        """Partition candidates deterministically within the context budget.

        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Deterministic candidate batches within the configured context budget.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        batches = []
        current: Candidates = {}
        entries = iter(candidates.items())
        entry = next(entries, None)
        while entry is not None:
            proposed = {**current, entry[0]: entry[1]}
            if not await self._fits(proposed):
                if current:
                    batches.append(current)
                current = {}
            current[entry[0]] = entry[1]
            if not await self._fits(current):
                raise ValueError(f"Candidate exceeds context: {entry[0]}")
            entry = next(entries, None)
        if current:
            batches.append(current)
        return batches

    async def _fits(self, candidates: Candidates, choice: bool = False) -> bool:
        """Check renderer character limits and optional exact tokenizer budgets.

        :param candidates: Proposed shared candidate batch.
        :param choice: Whether the batch is a choice shortlist.
        :returns: Whether the complete physical reads fit configured budgets.
        :raises ValueError: If exact tokens were requested from an unsupported backend.
        """
        if (
            self.context_size(candidates, choice)
            > self.engine.settings.max_context_chars
        ):
            return False
        limit = self.engine.settings.max_context_tokens
        if limit is None:
            return True
        count = await self.engine.decision_model.context_tokens(
            self._state(candidates), candidates, choice
        )
        if count is None:
            raise ValueError(
                "The selected decision model cannot enforce max_context_tokens"
            )
        return count <= limit

    async def _judge(self, candidates: Candidates) -> Scores:
        """Record a batch and validate its independent relevance scores.

        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Validated independent candidate relevance probabilities.
        """
        scores = await self.engine.decision_model.binary_judge(
            self._state(candidates), candidates
        )
        from odr.search_engine.engine import validate_scores

        validate_scores(scores, candidates)
        return scores

    def _state(self, candidates: Candidates) -> DecisionState:
        """Render a shared decision context from query, history and candidates.

        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Shared query and candidate context for a model request.
        """
        return {
            "query": self.query.text,
            "history": self.query.messages,
            "candidates": candidates,
        }

"""Public responses contracts."""

from __future__ import annotations

from odr.schema.records import FrozenModel


class SearchQuery(FrozenModel):
    """Query input deliberately excludes evaluation labels.

    :ivar text: Unmodified source or query text.
    :ivar messages: Ordered conversation history supplied with a query.
    """

    text: str
    messages: tuple[tuple[str, str], ...] = ()


class RetrievedPassage(FrozenModel):
    """Retrieval record with separate raw and ranking scores.

    :ivar node_id: Identity of the retrieved passage node.
    :ivar chunk_id: Original source chunk identity used by evaluation.
    :ivar document_id: Identity of the owning source document.
    :ivar text: Unmodified source or query text.
    :ivar path: Ancestor IDs from root to passage.
    :ivar rank: One-based rank in the returned result.
    :ivar binary_score: Independent unweighted passage relevance score.
    :ivar ranking_score: Score after the configured parent weighting policy.
    :ivar choice_score: Shortlist option probability when a choice was performed.
    :ivar metadata: Immutable source annotations.
    """

    node_id: str
    chunk_id: str = ""
    document_id: str
    text: str
    path: tuple[str, ...]
    rank: int
    binary_score: float
    ranking_score: float
    choice_score: float | None = None
    metadata: tuple[tuple[str, str], ...] = ()


class DecisionSearchRetrieve(FrozenModel):
    """Search output and numerical decision trace.

    :ivar query: Original validated query.
    :ivar passages: Ranked retrieved passages.
    :ivar corpus_revision: Content revision pinned for this retrieval.
    :ivar config_fingerprint: Semantic search configuration fingerprint.
    :ivar trace: Numerical traversal and candidate batch events.
    """

    query: SearchQuery
    passages: tuple[RetrievedPassage, ...]
    corpus_revision: str
    config_fingerprint: str
    trace: tuple[str, ...] = ()

    def to_text(self) -> str:
        """Render passages in rank order.

        :returns: Retrieved passage texts joined in rank order.
        """
        return "\n\n".join((passage.text for passage in self.passages))


class SearchEngineResponse(FrozenModel):
    """Optional generated answer with original retrieval.

    :ivar query: Original validated query.
    :ivar response: Generated answer text.
    :ivar retrieval: Original retrieval used for answer generation.
    """

    query: SearchQuery
    response: str
    retrieval: DecisionSearchRetrieve

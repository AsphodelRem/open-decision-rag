"""Dataset and metric contracts for gold-only evaluation."""

from typing import Literal

from typing_extensions import NotRequired, TypedDict


class EvaluationRecord(TypedDict):
    """Evaluation labels kept outside model requests.

    :ivar id: Unique source or record identity.
    :ivar query: Original validated query.
    :ivar qrels: Source chunk IDs mapped to relevance grades by default.
    :ivar id_space: Explicit source chunk or internal node identity namespace.
    """

    id: str
    query: str
    qrels: dict[str, float]
    id_space: NotRequired[Literal["chunk", "node"]]


class Metrics(TypedDict):
    """Metrics measured at the requested cutoff.

    :ivar hit: Indicator of any relevant result.
    :ivar recall: Fraction of relevant passages retrieved.
    :ivar ndcg: Normalized graded discounted gain.
    """

    hit: float
    recall: float
    ndcg: float


class EvaluationReport(Metrics):
    """Metrics associated with an evaluation query.

    :ivar id: Unique source or record identity.
    :ivar cutoff: Requested metric cutoff.
    :ivar requested_depth: Configured retrieval depth, at least the cutoff.
    :ivar returned_count: Number of passages actually returned.
    :ivar id_space: Source chunk or internal passage node label namespace.
    :ivar config_fingerprint: Semantic search configuration identity.
    :ivar corpus_revision: Evaluated immutable corpus revision.
    """

    id: str
    cutoff: int
    requested_depth: int
    returned_count: int
    id_space: str
    config_fingerprint: str
    corpus_revision: str

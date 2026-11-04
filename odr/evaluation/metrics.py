"""Gold-only retrieval metrics independent of model input schemas."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from odr.evaluation.types import EvaluationRecord, EvaluationReport, Metrics
    from odr.search_engine.engine import DecisionSearchEngine
import math


def discounted_gain(grades: Iterable[float]) -> float:
    """Accumulate graded gain with logarithmic rank discount.

    :param grades: Relevance grades in retrieval rank order.
    :returns: Total logarithmically discounted relevance gain.
    """
    contributions = []
    for rank, grade in enumerate(grades):
        gain = 2**grade - 1
        discount = math.log2(rank + 2)
        contributions.append(gain / discount)
    return sum(contributions)


def retrieval_metrics(
    retrieved: Iterable[str], relevance: Mapping[str, float], cutoff: int
) -> Metrics:
    """Compute distinct-ID Hit, Recall and graded nDCG at a cutoff.

    :param retrieved: Retrieved passage identities in rank order.
    :param relevance: Gold passage identities mapped to relevance grades.
    :param cutoff: Maximum retrieval rank included in metric computation.
    :returns: Hit, recall and normalized discounted gain at the cutoff.
    """
    selected = list(dict.fromkeys(retrieved))[:cutoff]
    (found, recall) = recall_score(selected, relevance)
    ideal_gain = discounted_gain(sorted(relevance.values(), reverse=True)[:cutoff])
    gain = discounted_gain((relevance.get(identity, 0) for identity in selected))
    return {
        "hit": float(bool(found)),
        "recall": recall,
        "ndcg": gain / ideal_gain if ideal_gain else 0,
    }


async def evaluate(
    engine: DecisionSearchEngine, records: Iterable[EvaluationRecord], cutoff: int = 10
) -> list[EvaluationReport]:
    """Keep labels out of requests and process conversations in input order.

    :param engine: Configured retrieval engine.
    :param records: Ordered records to read, rank, validate, or evaluate.
    :param cutoff: Maximum retrieval rank included in metric computation.
    :returns: Per-query gold-only evaluation reports in input order.
    """
    validate_evaluation(engine, cutoff)
    reports = []
    records = iter(records)
    record = next(records, None)
    while record is not None:
        reports.append(await evaluate_record(engine, record, cutoff))
        record = next(records, None)
    return reports


async def evaluate_record(
    engine: DecisionSearchEngine, record: EvaluationRecord, cutoff: int
) -> EvaluationReport:
    """Evaluate one query with explicit passage identity labels.

    :param engine: Configured retrieval engine.
    :param record: One evaluation query with explicit gold labels.
    :param cutoff: Maximum retrieval rank included in metric computation.
    :returns: Gold-only metrics associated with the query identity.
    """
    validate_evaluation(engine, cutoff)
    id_space = record.get("id_space", "chunk")
    if id_space not in {"chunk", "node"}:
        raise ValueError("Evaluation id_space must be chunk or node")
    await validate_qrels(engine, record, id_space)
    retrieval = await engine.search(record["query"])
    identities = [
        passage.node_id if id_space == "node" else passage.chunk_id
        for passage in retrieval.passages
    ]
    return {
        "id": record["id"],
        "cutoff": cutoff,
        "requested_depth": engine.settings.return_top_k,
        "returned_count": len(retrieval.passages),
        "id_space": id_space,
        "config_fingerprint": retrieval.config_fingerprint,
        "corpus_revision": retrieval.corpus_revision,
        **retrieval_metrics(identities, record["qrels"], cutoff),
    }


def recall_score(
    selected: Iterable[str], relevance: Mapping[str, float]
) -> tuple[set[str], float]:
    """Distinguish multi-gold recall from a hit indicator.

    :param selected: Distinct passage identities included in the cutoff.
    :param relevance: Gold passage identities mapped to relevance grades.
    :returns: Relevant retrieved IDs and the fraction of all gold IDs retrieved.
    """
    relevant = {identity for (identity, grade) in relevance.items() if grade > 0}
    found = relevant.intersection(selected)
    return (found, len(found) / len(relevant) if relevant else 0)


def validate_evaluation(engine: DecisionSearchEngine, cutoff: int) -> None:
    """Reject misleading evaluation depths before sending model requests.

    :param engine: Configured retrieval engine.
    :param cutoff: Requested metric cutoff.
    :raises ValueError: If the cutoff is invalid or exceeds configured retrieval depth.
    """
    if cutoff < 1 or engine.settings.return_top_k < cutoff:
        raise ValueError("Evaluation requires 1 <= cutoff <= engine.return_top_k")


async def validate_qrels(
    engine: DecisionSearchEngine, record: EvaluationRecord, id_space: str
) -> None:
    """Reject labels from a different identity namespace or corpus before search.

    :param engine: Revision-bound retrieval engine.
    :param record: Gold-only evaluation record.
    :param id_space: Explicit source chunk or internal passage node namespace.
    :raises ValueError: If a label does not identify a passage in the indexed corpus.
    """
    if id_space == "chunk":
        await engine.index.get_chunks(list(record["qrels"]))
        return
    nodes = await engine.index.get_nodes(list(record["qrels"]))
    if any(node is None or node.kind != "passage" for node in nodes):
        raise ValueError("Evaluation qrels contain missing or non-passage node IDs")

"""Exact-token logprob readout for raw vLLM DiffusionGemma responses."""

from __future__ import annotations

import math

from odr.models.types import WireResponse
from odr.models.vllm_types import CanvasPlan, LogprobRow, VLLMResponse


def token_evidence(row: LogprobRow) -> dict[int, float]:
    """Require unambiguous token IDs rather than decoded token spellings.

    :param row: Provider evidence at the answer slot.
    :returns: Exact token IDs mapped to finite log probabilities.
    :raises ValueError: If identities are malformed or duplicated.
    """
    evidence = {}
    for entry in row.top_logprobs:
        token_id = token_identity(entry.token)
        if token_id in evidence:
            raise ValueError("Duplicate token ID in vLLM logprobs")
        evidence[token_id] = entry.logprob
    return evidence


def read_probabilities(response: WireResponse, plan: CanvasPlan) -> list[float]:
    """Normalize exact evidence over all allowed labels using stable softmax.

    :param response: Complete cached or freshly received vLLM wire response.
    :param plan: Tokenizer-derived answer slot and ordered label IDs.
    :returns: Normalized label probabilities in the plan's original order.
    :raises ValueError: If rows, exact IDs or finite label evidence are missing.
    """
    completion = VLLMResponse.model_validate(response, strict=True)
    rows = completion.choices[0].logprobs.rows
    if plan.position >= len(rows):
        raise ValueError("vLLM response is missing the answer-slot row")
    evidence = token_evidence(rows[plan.position])
    if not set(plan.label_ids).issubset(evidence):
        raise ValueError("vLLM did not return every requested label logprob")
    logits = [evidence[token] for token in plan.label_ids]
    return normalized_weights(logits)


def normalized_weights(logits: list[float]) -> list[float]:
    """Apply numerically stable softmax without introducing missing-token floors.

    :param logits: Finite exact log probabilities of every allowed label.
    :returns: Ordered normalized probabilities.
    """
    largest = max(logits)
    weights = [math.exp(logit - largest) for logit in logits]
    total = sum(weights)
    return [weight / total for weight in weights]


def token_identity(spelling: str) -> int:
    """Parse an exact token identifier returned by the vLLM server.

    :param spelling: Provider token identity rather than its decoded text.
    :returns: Nonnegative tokenizer ID.
    :raises ValueError: If the token notation is unsupported.
    """
    prefix, separator, identity = spelling.partition(":")
    if prefix != "token_id" or not separator or not identity.isdecimal():
        raise ValueError("vLLM must return exact token_id identities")
    return int(identity)

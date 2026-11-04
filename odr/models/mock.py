"""Deterministic offline backends for examples and tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.models.types import Candidates, DecisionState, Messages, Scores, Vectors
import hashlib
import re

from odr.models.base import LLM, DecisionModel, Embedder

SUMMARY_TITLE_LIMIT = 20
BYTE_SCALE = 255


class MockDecisionModel(DecisionModel):
    """Lexical relevance with independent candidate probabilities."""

    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Score query words against each candidate.

        :param state: Query, history and all candidates sharing a decision context.
        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Independent relevance probabilities by candidate identity.
        """
        words = set(re.findall(r"\w+", state["query"].lower()))
        return {
            identity: len(words.intersection(re.findall(r"\w+", text.lower())))
            / max(len(words), 1)
            for (identity, text) in candidates.items()
        }

    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Normalize lexical scores for shortlist comparison.

        :param state: Query, history and all candidates sharing a decision context.
        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Normalized option probabilities by candidate identity.
        """
        scores = await self.binary_judge(state, candidates)
        total = sum(scores.values())
        return {
            identity: score / total if total else 1 / len(scores)
            for (identity, score) in scores.items()
        }


class MockEmbedder(Embedder):
    """Stable nonzero vectors computed without loading an embedding model."""

    async def embed(self, texts: Sequence[str]) -> Vectors:
        """Derive deterministic features from input bytes.

        :param texts: Ordered source texts to embed.
        :returns: Embedding vectors in input order.
        """
        return [
            [
                component / BYTE_SCALE
                for component in hashlib.sha256(text.encode()).digest()
            ]
            for text in texts
        ]


class MockLLM(LLM):
    """Deterministic valid summary for offline builds."""

    async def chat_completion(self, messages: Messages) -> str:
        """Return a short description including source titles.

        :param messages: Ordered role and content messages for generation.
        :returns: Generated response text.
        """
        words = messages[-1]["content"].split()
        return " ".join(words[-SUMMARY_TITLE_LIMIT:])

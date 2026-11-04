"""Small injectable model contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.models.types import Candidates, DecisionState, Messages, Scores, Vectors
import asyncio
import json
from abc import ABC, abstractmethod


class LLM(ABC):
    """Generation backend."""

    @abstractmethod
    async def chat_completion(self, messages: Messages) -> str:
        """Return generated text.

        :param messages: Ordered role and content messages for generation.
        :returns: Generated response text.
        """

    async def batch_chat_completion(self, batches: Sequence[Messages]) -> list[str]:
        """Generate each independent conversation.

        :param batches: Independent generation conversations.
        :returns: Generated texts in conversation input order.
        """
        return await asyncio.gather(
            *(self.chat_completion(messages) for messages in batches)
        )


class Embedder(ABC):
    """Ordered document vector backend."""

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> Vectors:
        """Return one finite vector per input text.

        :param texts: Ordered source texts to embed.
        :returns: Embedding vectors in input order.
        """


class DecisionModel(ABC):
    """Shared-context relevance and shortlist comparison."""

    def context_size(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int:
        """Estimate context characters for custom providers without a renderer hook.

        This fallback excludes provider templates and is explicitly approximate.
        Concrete HTTP adapters override it using their actual request renderers.

        :param state: Query and conversation history.
        :param candidates: Complete candidate context.
        :param choice: Whether to measure a choice request.
        :returns: Approximate serialized context size in Unicode characters.
        """
        return len(json.dumps({**state, "candidates": candidates}, ensure_ascii=False))

    async def context_tokens(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int | None:
        """Expose exact token budgeting when the backend owns a matching tokenizer.

        :param state: Query and conversation history.
        :param candidates: Complete candidate context.
        :param choice: Whether to measure a choice request.
        :returns: Exact token count, or None from a conditional custom renderer.
        :raises ValueError: If the backend has no exact tokenizer capability.
        """
        raise ValueError(
            "The selected decision model cannot enforce max_context_tokens"
        )

    @abstractmethod
    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Return independent yes probabilities keyed by candidate ID.

        :param state: Query, history and all candidates sharing a decision context.
        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Independent relevance probabilities by candidate identity.
        """

    @abstractmethod
    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Return an option distribution keyed by candidate ID.

        :param state: Query, history and all candidates sharing a decision context.
        :param candidates: Candidate identities mapped to rendered descriptions.
        :returns: Normalized option probabilities by candidate identity.
        """

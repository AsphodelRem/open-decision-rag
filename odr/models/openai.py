"""Cached OpenAI-compatible generation and embedding adapters."""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from odr.models.client import CachedAsyncOpenAI
    from odr.models.types import (
        Candidates,
        DecisionState,
        Messages,
        Scores,
        Vectors,
        WireResponse,
    )

from odr.models.base import LLM, DecisionModel, Embedder
from odr.models.decisions import chat_scores, decision_messages, message_size
from odr.models.embeddings import decode_embeddings
from odr.models.embeddings import validate_vectors as validate_vectors

EMBEDDING_BATCH_SIZE = 32


class LLMOpenAI(LLM, DecisionModel):
    """Generation and JSON relevance judging using an injected shared client."""

    def __init__(
        self, client: CachedAsyncOpenAI, model_name: str, **settings: Any
    ) -> None:
        """Bind generation to a shared cached client and model identity.

        :param client: Shared cached HTTP execution client.
        :param model_name: Model identity or served model alias.
        :param settings: Provider options forwarded to the selected backend.
        """
        self.client = client
        self.model_name = model_name
        self.settings = settings

    async def chat_completion(self, messages: Messages) -> str:
        """Return provider text while retaining full response in cache.

        :param messages: Ordered role and content messages for generation.
        :returns: Generated response text.
        """
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            **self.settings,
        }
        response = await self.client.request(
            "chat/completions", payload, validator=self._validate
        )
        return response["choices"][0]["message"]["content"]

    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Generate independent relevance estimates for all candidates.

        :param state: Query, history and shared candidate context.
        :param candidates: Candidate IDs mapped to rendered descriptions.
        :returns: Generated relevance probabilities by candidate ID.
        """
        return await self._judge(state, candidates)

    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Generate a normalized distribution over the complete shortlist.

        :param state: Query, history and shared candidate context.
        :param candidates: Candidate IDs mapped to rendered descriptions.
        :returns: Generated normalized candidate probabilities.
        """
        return await self._judge(state, candidates, choice=True)

    def context_size(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int:
        """Measure the same message content sent by JSON judging.

        :param state: Query and conversation history.
        :param candidates: Complete candidate context.
        :param choice: Whether to include normalization instructions.
        :returns: Rendered characters before the server-side chat template.
        """
        return message_size(decision_messages(state, candidates, choice))

    async def _judge(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> Scores:
        """Validate generated scores before storing successful wire responses.

        :param state: Shared query context.
        :param candidates: Expected candidate identities.
        :param choice: Whether probabilities must sum to one.
        :returns: Validated scores, or an empty mapping without inference.
        """
        if not candidates:
            return {}
        payload = {
            "temperature": 0,
            **self.settings,
            "model": self.model_name,
            "response_format": {"type": "json_object"},
            "messages": decision_messages(state, candidates, choice),
        }
        readout = partial(chat_scores, candidates=candidates, choice=choice)
        response = await self.client.request(
            "chat/completions", payload, context="json-judge-v1", validator=readout
        )
        return readout(response)

    @classmethod
    def _validate(cls, response: WireResponse) -> None:
        """Validate the required fields of a successful provider response.

        :param response: Successful provider response to validate or persist.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        if not isinstance(response["choices"][0]["message"]["content"], str):
            raise ValueError("Generation response has no text")


class EmbedderOpenAI(Embedder):
    """Bounded HTTP embedding batches with validation before caching."""

    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        *,
        batch_size: int = EMBEDDING_BATCH_SIZE,
        **settings: Any,
    ) -> None:
        """Bind embeddings to a shared cached client and a bounded batch size.

        :param client: Shared HTTP execution client.
        :param model_name: Served embedding model identity.
        :param batch_size: Maximum inputs in a physical embedding request.
        :param settings: Provider-specific embedding request settings.
        :raises ValueError: If batch size is not a positive integer.
        """
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, int)
            or batch_size < 1
        ):
            raise ValueError("Embedding batch_size must be a positive integer")
        self.client = client
        self.model_name = model_name
        self.batch_size = batch_size
        self.settings = settings

    async def embed(self, texts: Sequence[str]) -> Vectors:
        """Encode bounded batches while preserving original input order.

        :param texts: Ordered source texts.
        :returns: One finite vector per source text, or an empty list.
        """
        vectors: Vectors = []
        batches = iter(
            texts[index : index + self.batch_size]
            for index in range(0, len(texts), self.batch_size)
        )
        batch = next(batches, None)
        while batch is not None:
            vectors.extend(await self._embed_batch(batch))
            batch = next(batches, None)
        return validate_vectors(vectors, len(texts)) if texts else []

    async def _embed_batch(self, texts: Sequence[str]) -> Vectors:
        """Validate exact batch coverage before storing successful responses.

        :param texts: One bounded input batch.
        :returns: Ordered validated embedding vectors.
        """
        readout = partial(decode_embeddings, count=len(texts))
        payload = {"model": self.model_name, "input": list(texts), **self.settings}
        response = await self.client.request("embeddings", payload, validator=readout)
        return readout(response)

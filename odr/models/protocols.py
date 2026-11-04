"""Structural contracts for local embedding components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol

from odr.models.types import Messages


class LocalEncoder(Protocol):
    """Encode interface provided by sentence transformers."""

    def encode(self, sentences: list[str], **settings: Any) -> ArrayEncoding:
        """Encode source text using provider options.

        :param sentences: Ordered source texts.
        :param settings: Provider encoding options.
        :returns: Provider array of embeddings.
        """
        ...


class ArrayEncoding(Protocol):
    """Array conversion supplied by a local embedding provider."""

    def tolist(self) -> list[list[float]]:
        """Convert the embedding matrix to Python scalars.

        :returns: Ordered document vectors.
        """
        ...


class DecisionTokenizer(Protocol):
    """Minimal tokenizer boundary for direct vLLM canvas compilation."""

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        """Tokenize a template without adding conversation boundaries.

        :param text: Template or special-token spelling.
        :param add_special_tokens: Whether to insert automatic boundary tokens.
        :returns: Token IDs in source order.
        """
        ...

    def decode(self, token_ids: list[int], *, skip_special_tokens: bool = False) -> str:
        """Decode a special token to verify exact tokenizer compatibility.

        :param token_ids: Token IDs to decode.
        :param skip_special_tokens: Whether to remove special tokens.
        :returns: Decoded token spelling.
        """
        ...

    def apply_chat_template(
        self, messages: Messages, **settings: Any
    ) -> list[int] | Mapping[str, list[int]]:
        """Tokenize complete chat messages with the matching server template.

        :param messages: Actual provider input messages.
        :param settings: Tokenization and generation-prompt template options.
        :returns: Token IDs or a mapping containing input_ids.
        """
        ...

    def __len__(self) -> int:
        """Return the tokenizer vocabulary size including added tokens.

        :returns: Exclusive upper bound for random canvas token IDs.
        """
        ...

"""Replaceable splitters with deterministic source coordinates."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from odr.chunker.protocols import ExternalChunker
    from odr.schema.records import Document
import re
from abc import ABC, abstractmethod

from odr.chunker.types import Tokenizer, TokenSpan
from odr.schema import Chunk


class BaseChunker(ABC):
    """Transform source documents into ordered chunks."""

    @abstractmethod
    def split(self, documents: Sequence[Document]) -> list[Chunk]:
        """Return chunks retaining document identity.

        :param documents: Source documents in corpus reading order.
        :returns: Ordered source chunks retaining document ownership.
        """


class TokenChunker(BaseChunker):
    """Pack paragraphs using injected token spans; default counts words."""

    def __init__(
        self,
        max_tokens: int = 384,
        overlap_tokens: int = 48,
        tokenizer: Tokenizer | None = None,
    ) -> None:
        """Configure token budgets and the source-span tokenizer.

        :param max_tokens: Maximum token count per chunk.
        :param overlap_tokens: Token overlap between neighboring chunks.
        :param tokenizer: Callable returning source character spans for tokens.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        if not 0 <= overlap_tokens < max_tokens:
            raise ValueError("Expected 0 <= overlap_tokens < max_tokens")
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.tokenizer = tokenizer or self._word_spans

    def __repr__(self) -> str:
        """Render the semantic splitter revision and token budgets.

        :returns: Stable semantic splitter representation.
        """
        return f"TokenChunker-v1:{self.max_tokens}:{self.overlap_tokens}"

    def split(self, documents: Sequence[Document]) -> list[Chunk]:
        """Split with token overlap and paragraph-preferred boundaries.

        :param documents: Source documents in corpus reading order.
        :returns: Ordered source chunks retaining document ownership.
        """
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    @classmethod
    def _word_spans(cls, text: str) -> Iterable[TokenSpan]:
        """Yield source spans for whitespace-delimited tokens.

        :param text: Source text to inspect.
        :returns: Source character spans of whitespace-delimited tokens.
        """
        return (match.span() for match in re.finditer(r"\S+", text))

    def _split_document(self, document: Document) -> list[Chunk]:
        """Split one source document while retaining its ownership.

        :param document: Source document owning the text or chunks.
        :returns: Chunks belonging to the supplied document.
        """
        spans = list(self.tokenizer(document.text))
        start = 0
        chunks: list[Chunk] = []
        while start < len(spans):
            end = min(start + self.max_tokens, len(spans))
            if end < len(spans):
                end = self._paragraph_end(document.text, spans, start, end)
            offsets = token_offsets(spans, start, end)
            chunks.append(
                Chunk(
                    doc_id=document.id,
                    text=document.text[slice(*offsets)],
                    chunk_order_idx=len(chunks),
                    source_start=offsets[0],
                    source_end=offsets[1],
                    chunker_revision=repr(self),
                )
            )
            start = self._next_start(start, end) if end < len(spans) else end
        return chunks

    def _paragraph_end(
        self, text: str, spans: Sequence[TokenSpan], start: int, end: int
    ) -> int:
        """Prefer the last paragraph boundary within the token budget.

        :param text: Source text to inspect.
        :param spans: Token character ranges in source order.
        :param start: Inclusive start of the selected range.
        :param end: Exclusive end of the selected range.
        :returns: Preferred exclusive ending token index.
        """
        boundaries = [
            index
            for index in range(start + self.overlap_tokens + 1, end)
            if paragraph_gap(text, spans, index)
        ]
        return boundaries[-1] if boundaries else end

    def _next_start(self, start: int, end: int) -> int:
        """Advance the source position while retaining configured overlap.

        :param start: Inclusive start of the selected range.
        :param end: Exclusive end of the selected range.
        :returns: Next starting source or token index, advancing by at least one.
        """
        return max(start + 1, end - self.overlap_tokens)


class SimpleChunker(BaseChunker):
    """Character-budget splitter preferring sentence boundaries."""

    def __init__(self, max_chunk_size: int = 1000, overlap: int = 0) -> None:
        """Configure character budgets and source overlap.

        :param max_chunk_size: Maximum number of source characters per chunk.
        :param overlap: Source-character overlap between neighboring chunks.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        if not 0 <= overlap < max_chunk_size:
            raise ValueError("Invalid character overlap")
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap

    def split(self, documents: Sequence[Document]) -> list[Chunk]:
        """Return source slices with exact character offsets.

        :param documents: Source documents in corpus reading order.
        :returns: Ordered source chunks retaining document ownership.
        """
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Chunk]:
        """Split one source document while retaining its ownership.

        :param document: Source document owning the text or chunks.
        :returns: Chunks belonging to the supplied document.
        """
        chunks: list[Chunk] = []
        start = 0
        while start < len(document.text):
            end = min(start + self.max_chunk_size, len(document.text))
            boundary = document.text.rfind(". ", start + self.overlap + 1, end)
            if boundary >= 0 and end < len(document.text):
                end = boundary + 1
            if document.text[start:end].strip():
                chunks.append(
                    Chunk(
                        doc_id=document.id,
                        text=document.text[start:end],
                        chunk_order_idx=len(chunks),
                        source_start=start,
                        source_end=end,
                        chunker_revision="simple-v1",
                    )
                )
            if end == len(document.text):
                break
            start = self._next_start(start, end)
        return chunks

    def _next_start(self, start: int, end: int) -> int:
        """Advance the source position while retaining configured overlap.

        :param start: Inclusive start of the selected range.
        :param end: Exclusive end of the selected range.
        :returns: Next starting source or token index, advancing by at least one.
        """
        return max(start + 1, end - self.overlap)


class RaguChunkerAdapter(BaseChunker):
    """Wrap an external splitter without adopting passage_text-only IDs."""

    def __init__(self, chunker: ExternalChunker) -> None:
        """Wrap an external splitter while retaining document ownership.

        :param chunker: Injected splitter, the default splitter sentinel, or no-split mode.
        """
        self.chunker = chunker

    def split(self, documents: Sequence[Document]) -> list[Chunk]:
        """Unknown coordinates remain explicitly unknown.

        :param documents: Source documents in corpus reading order.
        :returns: Ordered source chunks retaining document ownership.
        """
        chunks: list[Chunk] = []
        for document in documents:
            chunks.extend(self._split_document(document))
        return chunks

    def _split_document(self, document: Document) -> list[Chunk]:
        """Split one source document while retaining its ownership.

        :param document: Source document owning the text or chunks.
        :returns: Chunks belonging to the supplied document.
        """
        chunks: list[Chunk] = []
        for position, original in enumerate(self.chunker.split([document.text])):
            chunks.append(
                Chunk(
                    doc_id=document.id,
                    text=getattr(original, "content"),
                    chunk_order_idx=position,
                    chunker_revision="ragu-v1",
                    metadata=(
                        ("original_chunk_id", original.id),
                        ("offset_provenance", "unavailable"),
                    ),
                )
            )
        return chunks


def token_offsets(spans: Sequence[TokenSpan], start: int, end: int) -> TokenSpan:
    """Coordinates covering the selected token range.

    :param spans: Token character ranges in source order.
    :param start: Inclusive start of the selected range.
    :param end: Exclusive end of the selected range.
    :returns: Inclusive start and exclusive end character coordinates.
    """
    return (spans[start][0], spans[end - 1][1])


def paragraph_gap(text: str, spans: Sequence[TokenSpan], index: int) -> bool:
    """Whether adjacent tokens have a paragraph separator.

    :param text: Source text to inspect.
    :param spans: Token character ranges in source order.
    :param index: Position of the adjacent token boundary.
    :returns: Whether the token boundary contains a paragraph separator.
    """
    previous_end = spans[index - 1][1]
    following_start = spans[index][0]
    gap = text[previous_end:following_start]
    return "\n\n" in gap

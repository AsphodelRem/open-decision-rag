"""Structural contracts accepted by the external chunker adapter."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol


class ExternalChunk(Protocol):
    """Chunk supplied by an external splitter.

    :ivar id: Unique source or record identity.
    :ivar content: Source text returned by the splitter.
    """

    @property
    def id(self) -> str:
        """Return the source identity.

        :returns: External chunk ID.
        """
        ...


class ExternalChunker(Protocol):
    """Structural interface accepted by the RAGU adapter."""

    def split(self, documents: list[str]) -> Sequence[ExternalChunk]:
        """Split external document inputs.

        :param documents: Ordered source texts.
        :returns: Ordered external chunks.
        """
        ...

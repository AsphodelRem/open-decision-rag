"""Structural contract for injectable tree and source storage."""

from collections.abc import Sequence
from typing import Protocol

from odr.schema import Chunk, CorpusSnapshot, HierarchyNode


class TreeStorage(Protocol):
    """Revision-bound storage boundary implemented without provider inheritance.

    :ivar revision: Validated content revision.
    :ivar root_id: Root node identity.
    """

    revision: str
    root_id: str

    async def children(self, parent_id: str) -> list[HierarchyNode]:
        """Read immediate children in stable order.

        :param parent_id: Parent identity.
        :returns: Ordered children.
        """
        ...

    async def get_nodes(self, identities: Sequence[str]) -> list[HierarchyNode | None]:
        """Resolve nodes in input order.

        :param identities: Requested node identities.
        :returns: Corresponding nodes or explicit missing entries.
        """
        ...

    async def get_chunks(self, identities: Sequence[str]) -> list[Chunk | None]:
        """Resolve source chunks in input order.

        :param identities: Requested chunk identities.
        :returns: Corresponding chunks or explicit missing entries.
        """
        ...

    async def export(self) -> CorpusSnapshot:
        """Materialize and validate the pinned corpus.

        :returns: Complete validated snapshot.
        """
        ...

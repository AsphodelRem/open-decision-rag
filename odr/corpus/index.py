"""Public storage facade independent of physical index format."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.schema.records import Chunk, CorpusSnapshot, HierarchyNode
    from odr.storage.protocols import TreeStorage
from odr.storage.tree import InMemoryTreeStorage


class HierarchyIndex:
    """Coordinate tree and source KV reads for one immutable revision."""

    def __init__(self, storage: TreeStorage) -> None:
        """Bind tree and source access to one storage revision.

        :param storage: Revision-bound tree and source storage.
        """
        self.storage = storage
        self.root_id = storage.root_id
        self.revision = storage.revision

    @classmethod
    def from_snapshot(cls, snapshot: CorpusSnapshot) -> HierarchyIndex:
        """Create a memory index from a validated snapshot.

        :param snapshot: Validated immutable corpus snapshot.
        :returns: A facade or index pinned to the snapshot revision.
        """
        return cls(InMemoryTreeStorage(snapshot))

    async def children(self, parent_id: str) -> list[HierarchyNode]:
        """Read ordered children.

        :param parent_id: Parent node identity.
        :returns: Immediate child nodes in stable reading order.
        """
        return await self.storage.children(parent_id)

    async def get_nodes(self, identities: Sequence[str]) -> list[HierarchyNode | None]:
        """Preserve requested node order.

        :param identities: Requested identities in output order.
        :returns: Requested nodes in input order, with missing nodes represented by None.
        """
        return await self.storage.get_nodes(identities)

    async def get_chunks(self, identities: Sequence[str]) -> list[Chunk]:
        """Require source records for all requested chunks.

        :param identities: Requested identities in output order.
        :returns: Requested source chunks in input order.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        chunks = await self.storage.get_chunks(identities)
        if any((chunk is None for chunk in chunks)):
            raise ValueError("Index integrity error: missing chunk")
        return [chunk for chunk in chunks if chunk is not None]

    async def path(self, node_id: str | None) -> tuple[str, ...]:
        """Read ancestry as IDs, from root to passage.

        :param node_id: Hierarchy node identity, or an absent starting node.
        :returns: Ancestor identities ordered from root to the requested node.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        path = []
        while node_id is not None:
            node = (await self.get_nodes([node_id]))[0]
            if node is None or node_id in path:
                raise ValueError("Index integrity error: invalid ancestry")
            path.append(node.id)
            node_id = node.parent_id
        return tuple(reversed(path))

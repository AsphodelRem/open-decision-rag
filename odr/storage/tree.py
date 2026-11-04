"""Memory and lazy SQLite storage for immutable revisions."""

from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

import asyncio
import json
import sqlite3
from contextlib import closing

from odr.schema import records, types
from odr.storage.records import SQLiteRecords
from odr.storage.sqlite import read_connection


class InMemoryTreeStorage:
    """Read-only ordered tree and chunk KV."""

    def __init__(self, snapshot: records.CorpusSnapshot) -> None:
        """Index one immutable corpus revision in memory.

        :param snapshot: Validated immutable corpus snapshot.
        """
        self.snapshot = snapshot
        self.revision = snapshot.revision
        self.root_id = snapshot.root_id
        self.node_records = {node.id: node for node in snapshot.nodes}
        self.chunk_records = {chunk.id: chunk for chunk in snapshot.chunks}
        self.children_records: dict[str, list[records.HierarchyNode]] = {}
        for node in snapshot.nodes:
            if node.parent_id is not None:
                self.children_records.setdefault(node.parent_id, []).append(node)
        order = attrgetter("position", "id")
        for children in self.children_records.values():
            children.sort(key=order)

    async def children(self, parent_id: str) -> list[records.HierarchyNode]:
        """Read immediate children in stable corpus order.

        :param parent_id: Parent node identity.
        :returns: Immediate child nodes in stable reading order.
        """
        return list(self.children_records.get(parent_id, []))

    async def export(self) -> records.CorpusSnapshot:
        """Return the pinned immutable in-memory snapshot.

        :returns: Complete validated snapshot.
        """
        return self.snapshot

    async def get_nodes(
        self, identities: Sequence[str]
    ) -> list[records.HierarchyNode | None]:
        """Preserve request order and missing entries.

        :param identities: Requested identities in output order.
        :returns: Requested nodes in input order, with missing nodes represented by None.
        """
        return await self._read_many(self.node_records, identities)

    async def get_chunks(self, identities: Sequence[str]) -> list[records.Chunk | None]:
        """Read source chunks without silently dropping missing IDs.

        :param identities: Requested identities in output order.
        :returns: Requested source chunks in input order.
        """
        return await self._read_many(self.chunk_records, identities)

    async def _read_many(
        self, records: Mapping[str, types.RecordT], identities: Sequence[str]
    ) -> list[types.RecordT | None]:
        """Read records in request order without dropping missing entries.

        :param records: Ordered records to read, rank, validate, or evaluate.
        :param identities: Requested identities in output order.
        :returns: Requested records in input order, with None for missing identities.
        """
        return [records.get(identity) for identity in identities]


class SQLiteTreeStorage(SQLiteRecords):
    """Read individual records without loading passage texts at open."""

    def __init__(self, path: str | Path) -> None:
        """Open the revision header without loading passage texts.

        :param path: Source or destination filesystem path.
        """
        self.path = str(path)
        with closing(read_connection(self.path)) as connection:
            header = connection.execute("SELECT payload FROM header").fetchone()
        self.header = json.loads(header[0])
        self.revision = self.header["revision"]
        self.root_id = self.header["root_id"]

    @classmethod
    async def create(
        cls, path: str | Path, snapshot: records.CorpusSnapshot
    ) -> SQLiteTreeStorage:
        """Persist a validated snapshot off the event loop.

        :param path: Source or destination filesystem path.
        :param snapshot: Validated immutable corpus snapshot.
        :returns: SQLite storage containing the supplied snapshot.
        """
        await asyncio.to_thread(cls._write, str(path), snapshot)
        return cls(path)

    @classmethod
    def create_sync(cls, path: str | Path, snapshot: records.CorpusSnapshot) -> None:
        """Write an index within an externally owned worker and writer lock.

        :param path: Unique staging path.
        :param snapshot: Validated snapshot to persist.
        """
        cls._write(path, snapshot)

    async def children(self, parent_id: str) -> list[records.HierarchyNode]:
        """Use the parent/order index for navigation.

        :param parent_id: Parent node identity.
        :returns: Immediate child nodes in stable reading order.
        """
        rows = await asyncio.to_thread(self._children, parent_id)
        return [records.HierarchyNode.model_validate_json(row[0]) for row in rows]

    async def get_nodes(
        self, identities: Sequence[str]
    ) -> list[records.HierarchyNode | None]:
        """Read requested nodes only.

        :param identities: Requested identities in output order.
        :returns: Requested nodes in input order, with missing nodes represented by None.
        """
        return await asyncio.to_thread(
            self._many, "nodes", identities, records.HierarchyNode
        )

    async def get_chunks(self, identities: Sequence[str]) -> list[records.Chunk | None]:
        """Read requested chunk texts only.

        :param identities: Requested identities in output order.
        :returns: Requested source chunks in input order.
        """
        return await asyncio.to_thread(self._many, "chunks", identities, records.Chunk)

    async def export(self) -> records.CorpusSnapshot:
        """Materialize an explicit JSON export, validating its checksum.

        :returns: The fully materialized and checksum-validated snapshot.
        """
        return await asyncio.to_thread(self._export)

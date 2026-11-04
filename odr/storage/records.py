"""Internal SQLiteRecords operations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeVar

from odr.schema.records import FrozenModel

if TYPE_CHECKING:
    from collections.abc import Sequence
import json
import sqlite3
from contextlib import closing

from odr.schema import CorpusSnapshot
from odr.storage.sqlite import read_connection

CHUNKS_TABLE = "chunks"
SQLITE_BATCH_SIZE = 500
SchemaT = TypeVar("SchemaT", bound=FrozenModel)


class SQLiteRecords:
    """Typed SQLite tree and source-record serialization operations.

    :ivar path: Ancestor IDs from root to passage.
    :ivar header: Decoded immutable SQLite revision header.
    """

    path: str
    header: dict[str, Any]

    def verify(self) -> None:
        """Validate legacy artifacts whose manifests predate byte checksums."""
        self._export()

    @classmethod
    def _write(cls, path: str | Path, snapshot: CorpusSnapshot) -> None:
        """Commit complete records in a short SQLite transaction.

        :param path: Source or destination filesystem path.
        :param snapshot: Validated immutable corpus snapshot.
        """
        with closing(sqlite3.connect(path)) as connection, connection:
            connection.execute("CREATE TABLE header (payload TEXT)")
            connection.execute(
                "CREATE TABLE nodes (id TEXT PRIMARY KEY,parent_id TEXT,position INTEGER,payload TEXT)"
            )
            connection.execute("CREATE INDEX children ON nodes(parent_id,position,id)")
            connection.execute("CREATE TABLE chunks (id TEXT PRIMARY KEY,payload TEXT)")
            connection.execute(
                "CREATE TABLE documents (id TEXT PRIMARY KEY,payload TEXT)"
            )
            header = snapshot.model_dump(exclude={"nodes", CHUNKS_TABLE, "documents"})
            connection.execute("INSERT INTO header VALUES (?)", (json.dumps(header),))
            connection.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?)",
                [
                    (node.id, node.parent_id, node.position, node.model_dump_json())
                    for node in snapshot.nodes
                ],
            )
            for table, records in (
                ("chunks", snapshot.chunks),
                ("documents", snapshot.documents),
            ):
                connection.executemany(
                    f"INSERT INTO {table} VALUES (?,?)",
                    [(record.id, record.model_dump_json()) for record in records],
                )

    def _children(self, parent_id: str) -> list[tuple[str]]:
        """Read serialized immediate children in corpus order.

        :param parent_id: Parent node identity.
        :returns: Serialized child records in stable reading order.
        """
        with closing(read_connection(self.path)) as connection:
            return connection.execute(
                "SELECT payload FROM nodes WHERE parent_id=? ORDER BY position,id",
                (parent_id,),
            ).fetchall()

    def _many(
        self, table: str, identities: Sequence[str], schema: type[SchemaT]
    ) -> list[SchemaT | None]:
        """Read and decode source records in the requested identity order.

        :param table: Internal source or tree table name.
        :param identities: Requested identities in output order.
        :param schema: Pydantic record type used to decode stored JSON.
        :returns: Requested decoded records in input order, with None for missing identities.
        """
        if table not in {"nodes", CHUNKS_TABLE}:
            raise ValueError("Unsupported record table")
        records: dict[str, SchemaT] = {}
        with closing(read_connection(self.path)) as connection:
            for start in range(0, len(identities), SQLITE_BATCH_SIZE):
                records.update(
                    self._batch(
                        connection,
                        table,
                        identities[start : start + SQLITE_BATCH_SIZE],
                        schema,
                    )
                )
        return [records.get(identity) for identity in identities]

    @classmethod
    def _batch(
        cls,
        connection: sqlite3.Connection,
        table: str,
        identities: Sequence[str],
        schema: type[SchemaT],
    ) -> dict[str, SchemaT]:
        """Read a bounded SQL IN batch while preserving duplicate IDs externally.

        :param connection: Explicitly owned read-only connection.
        :param table: Validated internal table name.
        :param identities: Bounded requested identities.
        :param schema: Decoding schema.
        :returns: Found records indexed by identity.
        """
        placeholders = ",".join("?" for identity in identities)
        rows = connection.execute(
            f"SELECT id,payload FROM {table} WHERE id IN ({placeholders})",
            tuple(identities),
        )
        return {
            identity: schema.model_validate_json(payload) for identity, payload in rows
        }

    def _export(self) -> CorpusSnapshot:
        """Materialize the tree and source records as a validated snapshot.

        :returns: The fully materialized and checksum-validated snapshot.
        """
        payload = dict(self.header)
        with closing(read_connection(self.path)) as connection:
            for table in ("nodes", CHUNKS_TABLE, "documents"):
                payload[table] = [
                    json.loads(row[0])
                    for row in connection.execute(f"SELECT payload FROM {table}")
                ]
        return CorpusSnapshot.model_validate(payload)

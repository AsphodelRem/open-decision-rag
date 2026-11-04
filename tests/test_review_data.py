"""Regression checks for snapshot complexity, revisions and publication integrity."""

from __future__ import annotations

import asyncio
import sqlite3
import tempfile
import unittest
from collections.abc import Iterator, Mapping
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

from odr import CorpusHierarchy, CorpusSnapshot, StorageArguments
from odr.schema import Chunk, Document, HierarchyNode, validation
from odr.storage import sqlite, tree

DOCUMENT_ID = "document"
ROOT_ID = "root"
PASSAGE_COUNT = 1000
WORK_MULTIPLIER = 8


def documents(count: int = 1) -> list[Document]:
    """Create a stable prepared corpus with explicit source chunk identities.

    :param count: Number of prepared source passages.
    :returns: One titled source document with ordered passages.
    """
    return [
        Document(
            id=DOCUMENT_ID,
            title="Manual",
            chunks=tuple(
                Chunk(
                    id=f"p{index}",
                    doc_id=DOCUMENT_ID,
                    text=f"Passage {index}",
                    chunk_order_idx=index,
                )
                for index in range(count)
            ),
        )
    ]


def snapshot(count: int = 1) -> CorpusSnapshot:
    """Construct a valid immutable tree without depending on HTTP models.

    :param count: Number of passage leaves.
    :returns: Validated source snapshot.
    """
    sources = documents(count)
    nodes = [
        HierarchyNode(id=ROOT_ID, kind=ROOT_ID, description="Corpus"),
        HierarchyNode(
            id=DOCUMENT_ID, kind="document", parent_id=ROOT_ID, description="Manual"
        ),
    ]
    nodes.extend(
        HierarchyNode(
            id=f"passage:{chunk.id}",
            kind="passage",
            parent_id=DOCUMENT_ID,
            position=chunk.chunk_order_idx,
            chunk_id=chunk.id,
        )
        for chunk in sources[0].chunks
    )
    return CorpusSnapshot(
        corpus_id="fixture",
        root_id=ROOT_ID,
        nodes=tuple(nodes),
        documents=tuple(sources),
        chunks=sources[0].chunks,
    )


class CountingNodes(Mapping[str, HierarchyNode]):
    """Count graph accesses to verify linear work independently of wall time."""

    def __init__(self, nodes: tuple[HierarchyNode, ...]) -> None:
        """Index nodes and initialize the access counter.

        :param nodes: Valid tree nodes.
        """
        self.records = {node.id: node for node in nodes}
        self.accesses = 0

    def __getitem__(self, identity: str) -> HierarchyNode:
        """Count a graph lookup.

        :param identity: Requested node identity.
        :returns: Matching node.
        """
        self.accesses += 1
        return self.records[identity]

    def __iter__(self) -> Iterator[str]:
        """Iterate node identities.

        :returns: Node identity iterator.
        """
        return iter(self.records)

    def __len__(self) -> int:
        """Return the graph size.

        :returns: Number of unique nodes.
        """
        return len(self.records)


class SnapshotTests(unittest.TestCase):
    """Constructor parity and graph validation invariants."""

    def test_constructor_parity(self) -> None:
        """Constructor, validation and JSON entry points compute the same revision."""
        original = snapshot()
        payload = original.model_dump(exclude={"revision"})
        direct = CorpusSnapshot(**payload)
        self.assertTrue(direct.revision)
        self.assertEqual(
            direct.model_dump(exclude_unset=True)["revision"], original.revision
        )
        self.assertEqual(direct, CorpusSnapshot.model_validate(payload))
        self.assertEqual(
            original, CorpusSnapshot.model_validate_json(direct.model_dump_json())
        )

    def test_linear_validation(self) -> None:
        """Each graph node is accessed only a bounded number of times."""
        corpus = snapshot(PASSAGE_COUNT)
        nodes = CountingNodes(corpus.nodes)
        sources = [
            {chunk.id: chunk for chunk in corpus.chunks},
            {document.id: document for document in corpus.documents},
        ]
        with patch.object(validation, "unique_records", side_effect=[nodes, *sources]):
            validation.validate_snapshot(corpus)
        self.assertLess(nodes.accesses, WORK_MULTIPLIER * len(nodes))

    def test_disconnected_cycle(self) -> None:
        """Disconnected cycles fail without recursive ancestry traversal."""
        corpus = snapshot()
        cyclic = tuple(
            HierarchyNode(
                id=identity, parent_id=parent, kind="group", description=identity
            )
            for identity, parent in (("a", "b"), ("b", "a"))
        )
        with self.assertRaises(ValueError):
            CorpusSnapshot.model_validate(
                {
                    **corpus.model_dump(exclude={"revision"}),
                    "nodes": (*corpus.nodes, *cyclic),
                }
            )


class PublicationTests(unittest.IsolatedAsyncioTestCase):
    """Concurrent writers, read-only connections and artifact checksum checks."""

    async def test_concurrent_publication(self) -> None:
        """Same-process writers use distinct staging files and one atomic manifest."""
        with tempfile.TemporaryDirectory() as folder:
            published = await asyncio.gather(
                *(
                    CorpusHierarchy(
                        storage_settings=StorageArguments(storage_folder=folder)
                    ).build_from_docs(documents())
                    for index in range(3)
                )
            )
            reopened = await CorpusHierarchy.open(folder)
            self.assertEqual(
                {
                    corpus.index.revision
                    for corpus in published
                    if corpus.index is not None
                },
                {reopened.index.revision if reopened.index else ""},
            )
            self.assertFalse(list(Path(folder).glob(".publish-*")))

    async def test_modified_artifact(self) -> None:
        """Changing persisted source text invalidates the artifact checksum."""
        with tempfile.TemporaryDirectory() as folder:
            await CorpusHierarchy(
                storage_settings=StorageArguments(storage_folder=folder)
            ).build_from_docs(documents())
            artifact = next(Path(folder).glob("*.sqlite3"))
            with closing(sqlite3.connect(artifact)) as connection:
                connection.execute(
                    "UPDATE chunks SET payload=replace(payload, 'Passage', 'Changed')"
                )
                connection.commit()
            with self.assertRaisesRegex(ValueError, "checksum"):
                await CorpusHierarchy.open(folder)

    async def test_memory_publication(self) -> None:
        """The memory backend persists JSON and reopens as in-memory storage."""
        with tempfile.TemporaryDirectory() as folder:
            await CorpusHierarchy(
                storage_settings=StorageArguments(
                    storage_folder=folder, tree_backend="memory"
                )
            ).build_from_docs(documents())
            reopened = await CorpusHierarchy.open(folder)
            assert reopened.index is not None
            self.assertIsInstance(reopened.index.storage, tree.InMemoryTreeStorage)

    async def test_readonly_connection(self) -> None:
        """Published SQLite readers cannot modify the source database."""
        with tempfile.TemporaryDirectory() as folder:
            await CorpusHierarchy(
                storage_settings=StorageArguments(storage_folder=folder)
            ).build_from_docs(documents())
            artifact = next(Path(folder).glob("*.sqlite3"))
            with closing(sqlite.read_connection(artifact)) as connection:
                with self.assertRaises(sqlite3.OperationalError):
                    connection.execute("DELETE FROM chunks")

    async def test_batched_reads(self) -> None:
        """A multi-ID lookup uses one SELECT and closes its connection explicitly."""
        with tempfile.TemporaryDirectory() as folder:
            await CorpusHierarchy(
                storage_settings=StorageArguments(storage_folder=folder)
            ).build_from_docs(documents())
            storage = tree.SQLiteTreeStorage(next(Path(folder).glob("*.sqlite3")))
            connection = sqlite3.connect(storage.path, check_same_thread=False)
            statements: list[str] = []
            connection.set_trace_callback(statements.append)
            with patch("odr.storage.records.read_connection", return_value=connection):
                found = await storage.get_nodes(
                    [ROOT_ID, DOCUMENT_ID, "missing", ROOT_ID]
                )
            self.assertEqual(found[0], found[-1])
            self.assertIsNone(found[2])
            self.assertEqual(len(statements), 1)
            with self.assertRaises(sqlite3.ProgrammingError):
                connection.execute("SELECT 1")

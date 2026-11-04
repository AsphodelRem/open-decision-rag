"""Index construction and source-preservation regression tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Literal
import asyncio
import unittest

from odr import (
    BuilderArguments,
    CorpusHierarchy,
    Document,
    MockEmbedder,
    MockLLM,
    SimpleChunker,
    TokenChunker,
)
from odr.evaluation.metrics import retrieval_metrics
from odr.storage.tree import InMemoryTreeStorage

FIRST_ID = "first"
CHARACTER_BUDGET = 15


class IndexingTests(unittest.TestCase):
    """Exercise real clustering and source offsets."""

    def test_clustering(self) -> None:
        """Both clustering builders preserve all document and passage IDs."""
        for mode in ("kmeans", "recursive"):
            asyncio.run(self._cluster(mode))

    def test_offsets(self) -> None:
        """Source slices and duplicate-document identity remain exact."""
        documents = [
            Document(id=identity, text="One paragraph.\n\nAnother paragraph here.")
            for identity in (FIRST_ID, "second")
        ]
        for chunker in (
            TokenChunker(max_tokens=3, overlap_tokens=1),
            SimpleChunker(max_chunk_size=CHARACTER_BUDGET, overlap=2),
        ):
            chunks = chunker.split(documents)
            self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
            for chunk in chunks:
                self.assertEqual(
                    chunk.text, documents[0].text[chunk.source_start : chunk.source_end]
                )

    def test_multi_gold_metrics(self) -> None:
        """One relevant hit is half recall for two gold passages."""
        metrics = retrieval_metrics([FIRST_ID], {FIRST_ID: 1, "second": 1}, 1)
        self.assertEqual(metrics["hit"], 1)
        self.assertEqual(metrics["recall"], 0.5)
        self.assertEqual(metrics["ndcg"], 1)

    async def _cluster(self, mode: Literal["kmeans", "recursive"]) -> None:
        """Verify document preservation through the selected clustering builder.

        :param mode: Clustering builder variant under test.
        """
        hierarchy = CorpusHierarchy(
            embedder=MockEmbedder(),
            llm=MockLLM(),
            builder_settings=BuilderArguments(
                hierarchy_type=mode, n_clusters=2, max_documents_per_group=1
            ),
        )
        await hierarchy.build_from_docs(
            [
                Document(
                    id=FIRST_ID, title="Installation", text="Install this program."
                ),
                Document(id="second", title="Billing", text="Pay invoices monthly."),
                Document(id="third", title="Settings", text="Configure this program."),
            ]
        )
        assert hierarchy.index is not None
        assert isinstance(hierarchy.index.storage, InMemoryTreeStorage)
        snapshot = hierarchy.index.storage.snapshot
        self.assertEqual(len(snapshot.documents), 3)
        self.assertEqual(len(snapshot.chunks), 3)
        documents = [node for node in snapshot.nodes if node.kind == "document"]
        self.assertEqual(len(documents), 3)

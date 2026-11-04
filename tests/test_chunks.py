"""Regression checks for model-owned chunk identity generation."""

from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

from odr import Chunk, Document, RaguChunkerAdapter, SimpleChunker, TokenChunker
from odr.builders.preparation import prepare_chunks
from odr.chunker.protocols import ExternalChunk

SOURCE_ID = "source"
SOURCE_TEXT = "Repeated text. Repeated text."
EXPLICIT_ID = "prepared-id"
CHARACTER_LIMIT = 15
NO_SPLIT_ID = "10640782d188dd5583469ded70fb292ca1b00f53136278dfb3003d82950c52d4"


class ChunkTests(unittest.TestCase):
    """Verify identity scope, compatibility and serialization behavior."""

    def test_identity_scope(self) -> None:
        """Distinct source positions and revisions retain distinct identities."""
        chunks = [
            Chunk(doc_id=SOURCE_ID, text=SOURCE_TEXT),
            Chunk(doc_id="other", text=SOURCE_TEXT),
            Chunk(doc_id=SOURCE_ID, text=SOURCE_TEXT, chunk_order_idx=1),
            Chunk(doc_id=SOURCE_ID, text="Different text."),
            Chunk(doc_id=SOURCE_ID, text=SOURCE_TEXT, chunker_revision="next"),
        ]
        self.assertEqual(len({chunk.id for chunk in chunks}), len(chunks))
        repeated = Chunk(doc_id=SOURCE_ID, text=SOURCE_TEXT)
        self.assertEqual(repeated.id, chunks[0].id)
        self.assertEqual(
            Chunk.model_validate({"doc_id": SOURCE_ID, "content": SOURCE_TEXT}).id,
            chunks[0].id,
        )

    def test_explicit_identity(self) -> None:
        """Explicit IDs are preserved regardless of the splitter revision."""
        for identity in (EXPLICIT_ID, ""):
            chunk = Chunk(
                id=identity,
                doc_id=SOURCE_ID,
                text=SOURCE_TEXT,
                chunker_revision="different",
            )
            self.assertEqual(chunk.id, identity)
            self.assertEqual(
                Chunk.model_validate_json(chunk.model_dump_json()).id, identity
            )

    def test_round_trip(self) -> None:
        """Generated identities survive exports without construction-only fields."""
        chunk = Chunk(doc_id=SOURCE_ID, text=SOURCE_TEXT, chunker_revision="no-split")
        self.assertEqual(chunk.id, NO_SPLIT_ID)
        self.assertEqual(chunk.model_dump(exclude_unset=True)["id"], NO_SPLIT_ID)
        self.assertNotIn("chunker_revision", chunk.model_dump())
        self.assertEqual(
            Chunk.model_validate_json(chunk.model_dump_json()).id, NO_SPLIT_ID
        )

    def test_splitter_compatibility(self) -> None:
        """Splitters retain identities produced before removing the factory."""
        document = Document(id=SOURCE_ID, text=SOURCE_TEXT)
        cases = (
            (
                SimpleChunker(max_chunk_size=CHARACTER_LIMIT),
                (
                    "f1ad5fd539f89ddb06e4ac1b96e1f19dd8a0eb0ccf2c165331fbc886cdcda34d",
                    "2cd30d87ea9a6c69460c0d000dbda6b629aa1f847e233b7ff070cd303857a4d1",
                ),
            ),
            (
                TokenChunker(max_tokens=3, overlap_tokens=1),
                (
                    "340caa7be9ecdffee67c4a241841292870b475972a752e99907fb4e30bc49915",
                    "0fb221b0b8c7c5e339ba5dedee8f0996bcfee8055b8ceb65bc81f33f49478a5c",
                ),
            ),
        )
        for splitter, expected in cases:
            self.assertEqual(
                tuple(chunk.id for chunk in splitter.split([document])), expected
            )

    def test_no_split_and_prepared(self) -> None:
        """Raw sources generate IDs while prepared chunks pass through unchanged."""
        prepared = Chunk(id=EXPLICIT_ID, doc_id="ready", text=SOURCE_TEXT)
        documents = [
            Document(id=SOURCE_ID, text=SOURCE_TEXT),
            Document(id="ready", chunks=(prepared,)),
        ]
        chunks = asyncio.run(prepare_chunks(None, documents))
        self.assertIs(chunks[0], prepared)
        self.assertEqual(chunks[1].id, NO_SPLIT_ID)
        self.assertEqual(chunks[1].source_end, len(SOURCE_TEXT))

    def test_external_adapter(self) -> None:
        """Adapter metadata and unknown offsets survive direct construction."""
        chunks = RaguChunkerAdapter(ExternalSplitter()).split(
            [
                Document(id=SOURCE_ID, text=SOURCE_TEXT),
                Document(id="another", text=SOURCE_TEXT),
            ]
        )
        self.assertNotEqual(chunks[0].id, chunks[1].id)
        self.assertIsNone(chunks[0].source_start)
        self.assertIsNone(chunks[0].source_end)
        self.assertEqual(dict(chunks[0].metadata)["original_chunk_id"], EXPLICIT_ID)


class ExternalSplitter:
    """Minimal structural splitter preserving its own external identity."""

    def split(self, documents: list[str]) -> list[ExternalChunk]:
        """Return an external chunk for each input source.

        :param documents: Ordered source texts supplied by the adapter.
        :returns: Source chunks exposing external IDs and content.
        """
        return [SimpleNamespace(id=EXPLICIT_ID, content=text) for text in documents]

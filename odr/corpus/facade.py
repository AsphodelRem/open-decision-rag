"""Build, publish and reopen immutable corpus revisions."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from typing_extensions import Unpack

if TYPE_CHECKING:
    from collections.abc import Sequence

import asyncio

from odr import builders, schema
from odr import chunker as splitters
from odr.builders.preparation import prepare_chunks
from odr.corpus import components, persistence
from odr.corpus.index import HierarchyIndex
from odr.schema.provenance import build_manifest


class CorpusHierarchy(persistence.CorpusPersistence):
    """RAGU-style facade with explicit component injection."""

    def __init__(
        self,
        chunker: splitters.BaseChunker | Literal["default"] | None = "default",
        builder: builders.BaseHierarchyBuilder | None = None,
        **injected: Unpack[components.CorpusComponents],
    ) -> None:
        """Configure injected corpus construction and persistence components.

        :param chunker: Injected splitter, the default splitter sentinel, or no-split mode.
        :param builder: Injected hierarchy builder, overriding configured construction.
        :param injected: Injected models and validated build and storage settings.
        """
        components.validate_components(injected)
        self.chunker = splitters.TokenChunker() if chunker == "default" else chunker
        self.builder = builder
        self.components = injected
        self.builder_settings = (
            injected.get("builder_settings") or schema.BuilderArguments()
        )
        self.storage_settings = (
            injected.get("storage_settings") or schema.StorageArguments()
        )
        self.index = None

    async def __aenter__(self) -> CorpusHierarchy:
        """Enter the component context.

        :returns: The opened object.
        """
        return self

    async def __aexit__(self, *exception: object) -> bool:
        """Exit the component context according to resource ownership.

        :param exception: Exception context or provider failure being handled.
        :returns: Whether to suppress the current exception.
        """
        return False

    async def build_from_docs(
        self, documents: Sequence[schema.Document], resume: bool = False
    ) -> CorpusHierarchy:
        """Prepare source chunks, build and validate before publication.

        :param documents: Source documents in corpus reading order.
        :param resume: Whether to resume checkpoints; True is rejected until recovery is implemented.
        :returns: This corpus facade after successful construction.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        if resume:
            raise NotImplementedError("Build checkpoint recovery is not implemented")
        if not documents:
            raise ValueError("Cannot build an empty corpus")
        chunks = await prepare_chunks(self.chunker, documents)
        builder = self.builder or self._builder()
        nodes = await builder.build(documents, chunks)
        manifest = build_manifest(
            dict(
                builder=builder,
                chunker=self.chunker,
                sources=schema.fingerprint(
                    [document.model_dump(mode="json") for document in documents]
                ),
            )
        )
        snapshot = schema.CorpusSnapshot.model_validate(
            {
                "corpus_id": schema.fingerprint(
                    [document.id for document in documents]
                ),
                "root_id": "root",
                "documents": documents,
                "chunks": chunks,
                "nodes": nodes,
                "build_fingerprint": schema.fingerprint(manifest),
                "manifest": manifest,
            }
        )
        self.index = HierarchyIndex.from_snapshot(snapshot)
        if self.storage_settings.storage_folder:
            await self._publish(snapshot)
        return self

    def build_from_docs_sync(
        self, documents: Sequence[schema.Document], resume: bool = False
    ) -> CorpusHierarchy:
        """Build from synchronous code without nesting event loops.

        :param documents: Source documents in corpus reading order.
        :param resume: Whether to resume checkpoints; True is rejected until recovery is implemented.
        :returns: This corpus facade after successful construction.
        """
        return asyncio.run(self.build_from_docs(documents, resume=resume))

    def _builder(self) -> builders.BaseHierarchyBuilder:
        """Construct the configured hierarchy builder.

        :returns: The configured hierarchy builder.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        mode = self.builder_settings.hierarchy_type
        if mode == "flat":
            return builders.FlatHierarchyBuilder()
        if mode == "metadata":
            return builders.MetadataHierarchyBuilder()
        embedder = self.components.get("embedder")
        llm = self.components.get("llm")
        if embedder is None or llm is None:
            raise ValueError("Clustering requires an embedder and description LLM")
        builder_type = (
            builders.KMeansHierarchyBuilder
            if mode == "kmeans"
            else builders.RecursiveKMeansHierarchyBuilder
        )
        return builder_type(
            embedder, builders.LLMDescriptionBuilder(llm), self.builder_settings
        )

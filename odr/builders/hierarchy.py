"""Document-level hierarchy construction with real KMeans clustering."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.builders import types as build_types
import asyncio
from abc import ABC, abstractmethod

from odr.builders.descriptions import LLMDescriptionBuilder
from odr.builders.preparation import canonical_nodes
from odr.schema import Chunk, Document, HierarchyNode, fingerprint

DEFAULT_ITERATIONS = 300
SQLITE_TIMEOUT = 30
ROOT_ID = "root"


class BaseHierarchyBuilder(ABC):
    """Injectable hierarchy construction contract."""

    @abstractmethod
    async def build(
        self, documents: Sequence[Document], chunks: Sequence[Chunk]
    ) -> list[HierarchyNode]:
        """Return a complete ordered tree.

        :param documents: Source documents in corpus reading order.
        :param chunks: Prepared source chunks preserving document identity.
        :returns: All nodes of the constructed hierarchy.
        """


def source_nodes(
    documents: Sequence[Document], chunks: Sequence[Chunk], parents: dict[str, str]
) -> list[HierarchyNode]:
    """Attach each document and its prepared chunks exactly once.

    :param documents: Source documents in corpus reading order.
    :param chunks: Prepared source chunks preserving document identity.
    :param parents: Mapping from document identities to parent group identities.
    :returns: Document and passage nodes in source order.
    """
    nodes = []
    for position, document in enumerate(documents):
        nodes.append(
            HierarchyNode(
                id=document.id,
                kind="document",
                parent_id=parents[document.id],
                position=position,
                description=document.title or document.id,
            )
        )
    for chunk in chunks:
        nodes.append(
            HierarchyNode(
                id=f"passage:{chunk.id}",
                kind="passage",
                parent_id=chunk.doc_id,
                position=chunk.chunk_order_idx,
                chunk_id=chunk.id,
            )
        )
    return nodes


def root_node() -> HierarchyNode:
    """Create the canonical corpus root.

    :returns: The canonical corpus root node.
    """
    return HierarchyNode(id=ROOT_ID, kind="root", description="Corpus")


class FlatHierarchyBuilder(BaseHierarchyBuilder):
    """Root, documents, passages."""

    async def build(
        self, documents: Sequence[Document], chunks: Sequence[Chunk]
    ) -> list[HierarchyNode]:
        """Preserve source order without model calls.

        :param documents: Source documents in corpus reading order.
        :param chunks: Prepared source chunks preserving document identity.
        :returns: All nodes of the constructed hierarchy.
        """
        parents = {document.id: ROOT_ID for document in documents}
        return [root_node(), *source_nodes(documents, chunks, parents)]


class MetadataHierarchyBuilder(BaseHierarchyBuilder):
    """Group documents using explicit ordered metadata paths."""

    async def build(
        self, documents: Sequence[Document], chunks: Sequence[Chunk]
    ) -> list[HierarchyNode]:
        """Reuse common path prefixes without content deduplication.

        :param documents: Source documents in corpus reading order.
        :param chunks: Prepared source chunks preserving document identity.
        :returns: All nodes of the constructed hierarchy.
        """
        nodes = {ROOT_ID: root_node()}
        parents = {
            document.id: self._path(document.group_path, nodes)
            for document in documents
        }
        return [*nodes.values(), *source_nodes(documents, chunks, parents)]

    def _path(self, segments: Sequence[str], nodes: dict[str, HierarchyNode]) -> str:
        """Create or reuse all metadata groups along a document path.

        :param segments: Ordered metadata path segments.
        :param nodes: Hierarchy nodes or their identity mapping.
        :returns: Identity of the final metadata group or corpus root.
        """
        parent = ROOT_ID
        for depth, segment in enumerate(segments):
            identity = f"group:{fingerprint(segments[: depth + 1])}"
            nodes.setdefault(
                identity,
                HierarchyNode(
                    id=identity,
                    kind="group",
                    parent_id=parent,
                    position=len(nodes),
                    description=segment,
                    metadata=(("provenance", "metadata"),),
                ),
            )
            parent = identity
        return parent


class KMeansHierarchyBuilder(BaseHierarchyBuilder):
    """Cluster documents and summarize titles ordered by centroid distance."""

    def __init__(
        self,
        embedder: build_types.Embedder,
        description_builder: LLMDescriptionBuilder,
        settings: build_types.BuilderArguments,
    ) -> None:
        """Configure embeddings, descriptions and clustering parameters.

        :param embedder: Injected document embedding model.
        :param description_builder: Validated group-description generator.
        :param settings: Provider options forwarded to the selected backend.
        """
        self.embedder = embedder
        self.description_builder = description_builder
        self.settings = settings

    async def build(
        self, documents: Sequence[Document], chunks: Sequence[Chunk]
    ) -> list[HierarchyNode]:
        """Build groups from document features rather than passage vectors.

        :param documents: Source documents in corpus reading order.
        :param chunks: Prepared source chunks preserving document identity.
        :returns: All nodes of the constructed hierarchy.
        """
        vectors = await self.embedder.embed(
            [self._input(document) for document in documents]
        )
        groups = await asyncio.to_thread(
            self._partition, list(range(len(documents))), vectors
        )
        nodes = [root_node()]
        parents: dict[str, str] = {}
        await self._groups((documents, vectors, nodes, parents), groups, ROOT_ID, 0)
        return canonical_nodes([*nodes, *source_nodes(documents, chunks, parents)])

    @classmethod
    def _identity(cls, documents: Sequence[Document], members: Sequence[int]) -> str:
        """Derive the stable group identity from sorted document identities.

        :param documents: Source documents in corpus reading order.
        :param members: Document row indices assigned to a group.
        :returns: Stable identity of the group member set.
        """
        identities = sorted((documents[index].id for index in members))
        return f"group:{fingerprint(identities)}"

    @classmethod
    def _input(cls, document: Document) -> str:
        """Compose the title and opening sentence used for document embeddings.

        :param document: Source document owning the text or chunks.
        :returns: Document title and deterministic opening sentence.
        """
        opening = document.text or " ".join((chunk.text for chunk in document.chunks))
        first_sentence = opening.split(".")[0]
        return f"{document.title}\n{first_sentence}"

    async def _groups(
        self,
        context: build_types.BuildContext,
        groups: Sequence[Sequence[int]],
        parent: str,
        depth: int,
    ) -> None:
        """Construct sibling groups concurrently within their parent.

        :param context: Semantic cache context or shared hierarchy construction state.
        :param groups: Document row indices grouped by cluster.
        :param parent: Parent group identity.
        :param depth: Current hierarchy depth.
        """
        await asyncio.gather(
            *(
                self._group(context, members, (parent, position), depth)
                for (position, members) in enumerate(groups)
            )
        )

    async def _group(
        self,
        context: build_types.BuildContext,
        members: Sequence[int],
        location: tuple[str, int],
        depth: int,
    ) -> None:
        """Construct a group, generate its description and assign its documents.

        :param context: Semantic cache context or shared hierarchy construction state.
        :param members: Document row indices assigned to a group.
        :param location: Parent identity and sibling position for a new group.
        :param depth: Current hierarchy depth.
        """
        identity = self._identity(context[0], members)
        documents = context[0]
        titles = [documents[index].title or documents[index].id for index in members]
        description = await self.description_builder.describe(titles)
        context[2].append(
            HierarchyNode(
                id=identity,
                kind="group",
                parent_id=location[0],
                position=location[1],
                description=description,
            )
        )
        for index in members:
            context[3][context[0][index].id] = identity

    def _partition(
        self, members: Sequence[int], vectors: build_types.Vectors
    ) -> list[list[int]]:
        """Cluster document rows through the numerical backend.

        :param members: Document row indices assigned to a group.
        :param vectors: Ordered document embedding vectors.
        :returns: Document row indices grouped by cluster.
        """
        from odr.builders.clustering import partition

        return partition(members, vectors, self.settings)


class RecursiveKMeansHierarchyBuilder(KMeansHierarchyBuilder):
    """Recursively partition oversized groups with explicit depth stops."""

    async def _group(
        self,
        context: build_types.BuildContext,
        members: Sequence[int],
        location: tuple[str, int],
        depth: int,
    ) -> None:
        """Construct a group, generate its description and assign its documents.

        :param context: Semantic cache context or shared hierarchy construction state.
        :param members: Document row indices assigned to a group.
        :param location: Parent identity and sibling position for a new group.
        :param depth: Current hierarchy depth.
        """
        await super()._group(context, members, location, depth)
        if depth + 1 >= self.settings.max_depth:
            return
        if len(members) <= self.settings.max_documents_per_group:
            return
        children = await asyncio.to_thread(
            self._recursive_partition, members, context[1]
        )
        if len(children) > 1:
            identity = self._identity(context[0], members)
            await self._groups(context, children, identity, depth + 1)

    def _recursive_partition(
        self, members: Sequence[int], vectors: build_types.Vectors
    ) -> list[list[int]]:
        """Honor the recursive branching factor independently of root clusters.

        :param members: Document indices assigned to the current recursive group.
        :param vectors: Ordered document embeddings.
        :returns: Deterministic child groups.
        """
        from odr.builders.clustering import partition

        settings = self.settings.model_copy(
            update={"n_clusters": self.settings.branching_factor}
        )
        return partition(members, vectors, settings)

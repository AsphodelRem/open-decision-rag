"""Batched source resolution and ancestry reuse within one retrieval run."""

from __future__ import annotations

import json
from collections.abc import Sequence

from odr.corpus.index import HierarchyIndex
from odr.schema import Chunk, HierarchyNode, RetrievedPassage


class ResolvedSources:
    """Cache only nodes and chunks touched by the current search."""

    def __init__(self, index: HierarchyIndex) -> None:
        """Create empty per-query resolution caches.

        :param index: Revision-bound index used by this search.
        """
        self.index = index
        self.nodes: dict[str, HierarchyNode] = {}
        self.chunks: dict[str, Chunk] = {}
        self.paths: dict[str, tuple[HierarchyNode, ...]] = {}
        self.document_scores: dict[str, float] = {}

    async def resolve(self, nodes: Sequence[HierarchyNode]) -> None:
        """Batch-read missing chunks and shared ancestors before scoring.

        :param nodes: Candidate nodes supplied by tree navigation.
        """
        self.nodes.update({node.id: node for node in nodes})
        await self._parents(nodes)
        identities = list(
            dict.fromkeys(
                node.chunk_id
                for node in nodes
                if node.chunk_id is not None and node.chunk_id not in self.chunks
            )
        )
        self.chunks.update(
            {chunk.id: chunk for chunk in await self.index.get_chunks(identities)}
        )

    def path(self, node_id: str) -> tuple[HierarchyNode, ...]:
        """Reuse a validated source path without issuing further SQL reads.

        :param node_id: Already resolved candidate identity.
        :returns: Ancestors from corpus root through the candidate.
        :raises ValueError: If ancestry contains a cycle.
        """
        if node_id not in self.paths:
            trail = []
            visited = set()
            identity: str | None = node_id
            while identity is not None:
                if identity in visited:
                    raise ValueError("Cycle in persisted ancestry")
                visited.add(identity)
                trail.append(self.nodes[identity])
                identity = self.nodes[identity].parent_id
            self.paths[node_id] = tuple(reversed(trail))
        return self.paths[node_id]

    def order(self, node: HierarchyNode) -> tuple[tuple[int, str], ...]:
        """Compute deterministic document and passage reading order.

        :param node: Resolved candidate node.
        :returns: Hierarchical source-order key.
        """
        path = self.path(node.id)
        positions = (
            position
            for position, ancestor in enumerate(path)
            if ancestor.kind == "document"
        )
        start = next(positions, 0)
        return tuple((ancestor.position, ancestor.id) for ancestor in path[start:])

    def render(self, node: HierarchyNode) -> str:
        """Preserve document titles and heading paths in passage decision context.

        :param node: Resolved tree candidate.
        :returns: Description or a source-aware passage representation.
        """
        if node.kind != "passage":
            return node.description
        chunk = self.chunks[node.chunk_id or ""]
        document = next(
            ancestor for ancestor in self.path(node.id) if ancestor.kind == "document"
        )
        return json.dumps(
            dict(
                document=document.description,
                heading_path=chunk.heading_path,
                text=chunk.text,
            ),
            ensure_ascii=False,
        )

    def retrieved(
        self, node: HierarchyNode, score: float, scoring: str
    ) -> RetrievedPassage:
        """Assemble result metadata from records already loaded for scoring.

        :param node: Passage node.
        :param score: Native binary passage relevance.
        :param scoring: Binary or document-parent product ranking policy.
        :returns: Source-identified passage with stable ancestry.
        """
        path = self.path(node.id)
        document = next(ancestor for ancestor in path if ancestor.kind == "document")
        chunk = self.chunks[node.chunk_id or ""]
        weighted = (
            score * self.document_scores[document.id]
            if scoring == "parent_product"
            else score
        )
        return RetrievedPassage(
            node_id=node.id,
            chunk_id=chunk.id,
            document_id=document.id,
            text=chunk.text,
            path=tuple(ancestor.id for ancestor in path),
            rank=0,
            binary_score=score,
            ranking_score=weighted,
            metadata=chunk.metadata,
        )

    async def _parents(self, nodes: Sequence[HierarchyNode]) -> None:
        """Resolve each shared ancestor once using batched node reads.

        :param nodes: Nodes whose parent chains may not yet be cached.
        :raises ValueError: If persisted ancestry is incomplete.
        """
        pending = {
            node.parent_id
            for node in nodes
            if node.parent_id is not None and node.parent_id not in self.nodes
        }
        while pending:
            rows = await self.index.get_nodes(list(pending))
            if any(row is None for row in rows):
                raise ValueError("Index integrity error: missing ancestor")
            self.nodes.update({row.id: row for row in rows if row is not None})
            pending = {
                row.parent_id
                for row in rows
                if row is not None
                and row.parent_id is not None
                and row.parent_id not in self.nodes
            }

"""Source chunk preparation."""

from __future__ import annotations

from operator import attrgetter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.chunker.splitters import BaseChunker
    from odr.schema.records import Document
import asyncio

from odr.schema.records import Chunk, HierarchyNode


async def prepare_chunks(
    chunker: BaseChunker | None, documents: Sequence[Document]
) -> list[Chunk]:
    """Preserve prepared chunks and split only raw source documents.

    :param chunker: Injected splitter, the default splitter sentinel, or no-split mode.
    :param documents: Source documents in corpus reading order.
    :returns: Prepared and newly split chunks without content deduplication.
    """
    prepared = [chunk for document in documents for chunk in document.chunks]
    raw = [document for document in documents if not document.chunks]
    if chunker is None:
        split = [
            Chunk(
                doc_id=document.id,
                text=document.text,
                chunk_order_idx=0,
                source_start=0,
                source_end=len(document.text),
                chunker_revision="no-split",
            )
            for document in raw
        ]
    else:
        split = await asyncio.to_thread(chunker.split, raw)
    return [*prepared, *split]


def canonical_nodes(nodes: Sequence[HierarchyNode]) -> list[HierarchyNode]:
    """Serialize tree nodes in deterministic source reading order.

    :param nodes: Complete tree in any asynchronous completion order.
    :returns: Parent-first nodes ordered by sibling position and identity.
    """
    children = ordered_children(nodes)
    ordered = []
    pending = list(reversed(children.get(None, [])))
    while pending:
        node = pending.pop()
        ordered.append(node)
        pending.extend(reversed(children.get(node.id, [])))
    if len(ordered) != len(nodes):
        raise ValueError("Cannot serialize a disconnected tree")
    return ordered


def ordered_children(
    nodes: Sequence[HierarchyNode],
) -> dict[str | None, list[HierarchyNode]]:
    """Index siblings once and order them by source position.

    :param nodes: Complete hierarchy nodes.
    :returns: Deterministically ordered adjacency lists.
    """
    children: dict[str | None, list[HierarchyNode]] = {}
    for record in nodes:
        children.setdefault(record.parent_id, []).append(record)
    order = attrgetter("position", "id")
    for siblings in children.values():
        siblings.sort(key=order)
    return children

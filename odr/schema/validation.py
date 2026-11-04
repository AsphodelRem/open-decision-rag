"""Linear snapshot validation independent of storage."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from odr.schema.records import CorpusSnapshot, HierarchyNode
from odr.schema.types import RecordT, SourceRecords

NodeOwnership = tuple[str, str | None]
Adjacency = Mapping[str, list[str]]


def unique_records(records: Sequence[RecordT]) -> dict[str, RecordT]:
    """Index records without silently overwriting duplicates.

    :param records: Ordered source records.
    :returns: Records indexed by identity.
    :raises ValueError: If identities are duplicated.
    """
    indexed = {record.id: record for record in records}
    if len(indexed) != len(records):
        raise ValueError("Duplicate IDs")
    return indexed


def adjacency(nodes: Mapping[str, HierarchyNode]) -> dict[str, list[str]]:
    """Index parent references in one pass.

    :param nodes: Unique tree nodes.
    :returns: Parent identities mapped to immediate child identities.
    :raises ValueError: If a parent reference is dangling.
    """
    children: dict[str, list[str]] = defaultdict(list)
    for node in nodes.values():
        if node.parent_id is not None:
            if node.parent_id not in nodes:
                raise ValueError("Missing parent")
            children[node.parent_id].append(node.id)
    return children


def document_owners(
    nodes: Mapping[str, HierarchyNode], children: Adjacency, root: str
) -> dict[str, str | None]:
    """Traverse iteratively and propagate source ownership without rescanning paths.

    :param nodes: Unique tree nodes.
    :param children: Precomputed adjacency index.
    :param root: Sole validated root identity.
    :returns: Owning document for every reachable node.
    :raises ValueError: If the graph contains a cycle or disconnected component.
    """
    owners: dict[str, str | None] = {}
    pending: list[NodeOwnership] = [(root, None)]
    while pending:
        identity, owner = pending.pop()
        if identity in owners:
            raise ValueError("Cycle in hierarchy")
        if owner is None and nodes[identity].kind == "document":
            owner = identity
        owners[identity] = owner
        pending.extend((child, owner) for child in children.get(identity, []))
    if len(owners) != len(nodes):
        raise ValueError("Disconnected nodes or cycle in hierarchy")
    return owners


def validate_snapshot(snapshot: CorpusSnapshot) -> None:
    """Validate roots, references, cycles and source ownership in linear time.

    :param snapshot: Corpus snapshot under validation.
    :raises ValueError: If structural invariants are violated.
    """
    nodes = unique_records(snapshot.nodes)
    sources = (unique_records(snapshot.chunks), unique_records(snapshot.documents))
    check_root(nodes, snapshot.root_id)
    children = adjacency(nodes)
    owners = document_owners(nodes, children, snapshot.root_id)
    for node in nodes.values():
        _validate_node(node, owners, children, sources)


def _validate_node(
    node: HierarchyNode,
    owners: Mapping[str, str | None],
    children: Adjacency,
    sources: SourceRecords,
) -> None:
    """Check node contents against precomputed ancestry and adjacency.

    :param node: Tree node under validation.
    :param owners: Precomputed owning document identities.
    :param children: Precomputed immediate child identity index.
    :param sources: Chunk and document identity maps.
    :raises ValueError: If the node is inconsistent with its sources.
    """
    has_children = node.id in children
    if node.kind == "passage":
        _validate_passage(node, owners[node.id], sources, has_children)
    elif not node.description or not has_children:
        raise ValueError("Internal nodes require descriptions and children")


def _validate_passage(
    node: HierarchyNode,
    document_id: str | None,
    sources: SourceRecords,
    has_children: bool,
) -> None:
    """Require a terminal chunk belonging to the ancestor document.

    :param node: Passage node.
    :param document_id: Expected source document identity.
    :param sources: Chunk and document identity maps.
    :param has_children: Whether the passage has invalid child nodes.
    :raises ValueError: If the passage does not belong to its source document.
    """
    chunks, documents = sources
    chunk = None if node.chunk_id is None else chunks.get(node.chunk_id)
    if chunk is None or has_children:
        raise ValueError("Invalid passage reference or children")
    if document_id not in documents or chunk.doc_id != document_id:
        raise ValueError("Passage must belong to its source document")


def check_root(nodes: Mapping[str, HierarchyNode], root: str) -> None:
    """Require a single root of the declared kind.

    :param nodes: Unique node identity map.
    :param root: Declared root identity.
    :raises ValueError: If roots do not match the declaration.
    """
    roots = [node.id for node in nodes.values() if node.parent_id is None]
    if roots != [root] or nodes[root].kind != "root":
        raise ValueError("Expected exactly one root")

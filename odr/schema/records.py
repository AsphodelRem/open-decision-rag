"""Public records contracts."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from typing import Any
import hashlib
import json
from typing import Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


def fingerprint(payload: object) -> str:
    """Hash canonical JSON without timestamps or process-specific values.

    :param payload: Semantic operation input used for serialization or transport.
    :returns: Hexadecimal SHA-256 of canonical JSON.
    """
    encoded = json.dumps(
        payload, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


class FrozenModel(BaseModel):
    """Base for immutable public records."""

    def __getattr__(self, name: str) -> Any:
        """Expose declared serialization aliases as read-only attributes.

        :param name: Requested model attribute name or serialization alias.
        :returns: The stored value for the requested attribute or alias.
        """
        for field_name, field in type(self).model_fields.items():
            if field.alias == name:
                return getattr(self, field_name)
        return getattr(super(), "__getattr__")(name)

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class Chunk(FrozenModel):
    """Text unit with source coordinates in characters.

    :ivar id: Unique source or record identity.
    :ivar doc_id: Identity of the owning source document.
    :ivar text: Unmodified source or query text.
    :ivar chunk_order_idx: Zero-based chunk position within its document.
    :ivar num_tokens: Token count when supplied by the splitter.
    :ivar heading_path: Ordered source heading names.
    :ivar source_start: Inclusive character offset in source text.
    :ivar source_end: Exclusive character offset in source text.
    :ivar metadata: Immutable source annotations.
    :ivar chunker_revision: Splitter identity used only when generating a missing ID.
    """

    id: str = ""
    doc_id: str
    text: str = Field(min_length=1, alias="content")
    chunk_order_idx: int = 0
    num_tokens: int | None = None
    heading_path: tuple[str, ...] = ()
    source_start: int | None = None
    source_end: int | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    chunker_revision: str = Field(default="chunk-v1", exclude=True, repr=False)

    def model_post_init(self, context: object, /) -> None:
        """Generate a missing identity after source fields have been validated.

        Explicit IDs, including those loaded from snapshots, remain authoritative.
        Splitter revisions are construction inputs and are excluded from exports.

        :param context: Validation context supplied by Pydantic.
        """
        if "id" not in self.model_fields_set:
            identity = fingerprint(
                [
                    self.doc_id,
                    self.chunk_order_idx,
                    self.text,
                    self.chunker_revision,
                ]
            )
            object.__setattr__(self, "id", identity)
            self.model_fields_set.add("id")


class Document(FrozenModel):
    """Original document or prepared passage sequence.

    :ivar id: Unique source or record identity.
    :ivar title: Human-readable document title.
    :ivar text: Unmodified source or query text.
    :ivar chunks: Prepared source chunks retaining document identity.
    :ivar source_uri: Stable source URI or relative path.
    :ivar metadata: Immutable source annotations.
    :ivar group_path: Ordered metadata group names.
    """

    id: str
    title: str = ""
    text: str = ""
    chunks: tuple[Chunk, ...] = Field(
        default=(), validation_alias=AliasChoices("chunks", "passages")
    )
    source_uri: str = ""
    metadata: tuple[tuple[str, str], ...] = ()
    group_path: tuple[str, ...] = ()


class HierarchyNode(FrozenModel):
    """Ordered tree node; text belongs to the chunk store.

    :ivar id: Unique source or record identity.
    :ivar kind: Node role in the document hierarchy.
    :ivar parent_id: Parent node identity, absent for the root.
    :ivar position: Sibling order, with identity breaking ties.
    :ivar description: Description used for internal navigation.
    :ivar chunk_id: Identity of the source chunk referenced by a passage.
    :ivar metadata: Immutable source annotations.
    """

    id: str
    kind: Literal["root", "group", "document", "section", "passage"]
    parent_id: str | None = None
    position: int = 0
    description: str = ""
    chunk_id: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()


class CorpusSnapshot(FrozenModel):
    """Portable, content-addressed tree and source records.

    :ivar schema_version: Snapshot serialization schema version.
    :ivar corpus_id: Stable corpus identity.
    :ivar revision: Content fingerprint of the canonical snapshot.
    :ivar root_id: Identity of the sole corpus root.
    :ivar nodes: Immutable tree nodes.
    :ivar documents: Original source document records.
    :ivar chunks: Prepared source chunks retaining document identity.
    :ivar build_fingerprint: Semantic build configuration fingerprint.
    :ivar manifest: Immutable preprocessing provenance.
    """

    schema_version: Literal[1] = 1
    corpus_id: str
    revision: str = ""
    root_id: str
    nodes: tuple[HierarchyNode, ...]
    documents: tuple[Document, ...]
    chunks: tuple[Chunk, ...]
    build_fingerprint: str = ""
    manifest: tuple[tuple[str, str], ...] = ()

    @model_validator(mode="after")
    def validate_tree(self) -> CorpusSnapshot:
        """Reject dangling references, cycles and inconsistent ownership.

        :returns: The snapshot with its validated content revision.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        from odr.schema.validation import validate_snapshot

        validate_snapshot(self)
        payload = self.model_dump(exclude={"revision"})
        revision = fingerprint(payload)
        if self.revision and self.revision != revision:
            raise ValueError("Snapshot checksum mismatch")
        object.__setattr__(self, "revision", revision)
        self.model_fields_set.add("revision")
        return self

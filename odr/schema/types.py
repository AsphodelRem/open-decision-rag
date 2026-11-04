"""Source identity mappings shared by validation and storage."""

from typing import TypeVar

from odr.schema.records import Chunk, Document, HierarchyNode

ChunkRecords = dict[str, Chunk]
DocumentRecords = dict[str, Document]
SourceRecords = tuple[ChunkRecords, DocumentRecords]
RecordT = TypeVar("RecordT", Chunk, Document, HierarchyNode)

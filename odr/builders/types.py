"""Typed state passed through document hierarchy construction."""

from collections.abc import Sequence

from odr.models.base import Embedder as Embedder
from odr.models.types import Vectors as Vectors
from odr.schema.arguments import BuilderArguments as BuilderArguments
from odr.schema.records import Document, HierarchyNode

DocumentSequence = Sequence[Document]
NodeList = list[HierarchyNode]
DocumentParents = dict[str, str]
BuildContext = tuple[DocumentSequence, Vectors, NodeList, DocumentParents]

"""Deterministic local text and prepared JSON corpus ingestion."""

from __future__ import annotations

import json
from pathlib import Path

from odr.schema import Document, fingerprint


def load_documents(path: str | Path) -> list[Document]:
    """Load a directory of UTF-8 sources, JSON, or JSONL.

    :param path: Source or destination filesystem path.
    :returns: Source documents in deterministic input order.
    """
    source = Path(path)
    if source.is_dir():
        return [
            _text_document(source, file_path)
            for file_path in sorted(source.rglob("*"))
            if file_path.suffix.lower() in {".txt", ".md"}
        ]
    if source.suffix == ".jsonl":
        records = [
            json.loads(line) for line in source.read_text().splitlines() if line.strip()
        ]
    else:
        records = json.loads(source.read_text())
    return [Document.model_validate(record) for record in records]


def _text_document(root: Path, path: Path) -> Document:
    """Load source text and derive its stable relative-URI identity.

    :param root: Root directory used to derive stable source URIs.
    :param path: Source or destination filesystem path.
    :returns: A source document with a stable relative-URI identity.
    """
    uri = path.relative_to(root).as_posix()
    return Document(
        id=fingerprint(uri), title=path.stem, text=path.read_text(), source_uri=uri
    )

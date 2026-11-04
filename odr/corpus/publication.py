"""Locked atomic corpus publication and checksum-verified artifact opening."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from filelock import FileLock

from odr.schema import CorpusSnapshot
from odr.storage.protocols import TreeStorage
from odr.storage.tree import InMemoryTreeStorage, SQLiteTreeStorage

HASH_BLOCK_SIZE = 1048576
WRITER_TIMEOUT = 30
TEXT_ENCODING = "utf-8"
SQLITE_BACKEND = "sqlite"


def file_checksum(path: Path) -> str:
    """Hash an artifact incrementally without materializing source records.

    :param path: Existing artifact file.
    :returns: SHA-256 of the exact published bytes.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        block = source.read(HASH_BLOCK_SIZE)
        while block:
            digest.update(block)
            block = source.read(HASH_BLOCK_SIZE)
    return digest.hexdigest()


def open_artifact(path: Path, backend: str) -> TreeStorage:
    """Open a published backend using its explicit physical format.

    :param path: Existing artifact file.
    :param backend: Validated memory or SQLite backend name.
    :returns: Read-only revision-bound storage.
    :raises ValueError: If the backend is unknown.
    """
    if backend == "memory":
        return InMemoryTreeStorage(
            CorpusSnapshot.model_validate_json(path.read_text(encoding=TEXT_ENCODING))
        )
    if backend == SQLITE_BACKEND:
        return SQLiteTreeStorage(path)
    raise ValueError(f"Unsupported tree backend: {backend}")


def open_published(folder: Path) -> TreeStorage:
    """Verify artifact integrity before exposing the claimed content revision.

    :param folder: Published corpus bundle.
    :returns: Storage whose artifact checksum and revision have been verified.
    :raises ValueError: If metadata or artifact integrity checks fail.
    """
    manifest = json.loads((folder / "manifest.json").read_text(encoding=TEXT_ENCODING))
    path = folder / manifest["file"]
    checksum = manifest.get("sha256")
    if checksum is not None and file_checksum(path) != checksum:
        raise ValueError("Corpus artifact checksum mismatch")
    storage = open_artifact(path, manifest.get("backend", SQLITE_BACKEND))
    if checksum is None and isinstance(storage, SQLiteTreeStorage):
        storage.verify()
    if storage.revision != manifest["revision"]:
        raise ValueError("Manifest revision mismatch")
    return storage


def publish_snapshot(
    folder: Path, snapshot: CorpusSnapshot, backend: str
) -> TreeStorage:
    """Serialize writers across tasks and processes, using unique staging paths.

    :param folder: Destination corpus bundle directory.
    :param snapshot: Complete validated content revision.
    :param backend: Physical publication backend.
    :returns: Storage pinned to the revision published by this writer.
    """
    folder.mkdir(parents=True, exist_ok=True)
    with FileLock(str(folder / ".publish.lock"), timeout=WRITER_TIMEOUT):
        with tempfile.TemporaryDirectory(dir=folder, prefix=".publish-") as staging:
            artifact = write_artifact(Path(staging), snapshot, backend)
            checksum = file_checksum(artifact)
            target = folder / artifact.name
            if not target.exists() or file_checksum(target) != checksum:
                os.replace(artifact, target)
            write_manifest(folder, Path(staging), target, checksum, snapshot.revision)
            return open_artifact(target, backend)


def write_artifact(folder: Path, snapshot: CorpusSnapshot, backend: str) -> Path:
    """Write and close one complete artifact in a private staging directory.

    :param folder: Unique staging directory owned by one writer.
    :param snapshot: Validated snapshot to serialize.
    :param backend: Physical storage backend.
    :returns: Closed artifact ready for hashing and atomic replacement.
    """
    target = folder / f"{snapshot.revision}.sqlite3"
    if backend == "memory":
        target = target.with_suffix(".json")
        target.write_text(snapshot.model_dump_json(), encoding=TEXT_ENCODING)
    elif backend == SQLITE_BACKEND:
        SQLiteTreeStorage.create_sync(target, snapshot)
    else:
        raise ValueError(f"Unsupported tree backend: {backend}")
    return target


def write_manifest(
    folder: Path, staging: Path, artifact: Path, checksum: str, revision: str
) -> None:
    """Atomically publish metadata only after the complete artifact exists.

    :param folder: Destination corpus bundle.
    :param staging: Writer-specific temporary directory.
    :param artifact: Complete published artifact.
    :param checksum: Exact artifact SHA-256.
    :param revision: Semantic corpus revision.
    """
    manifest: dict[str, Any] = dict(
        revision=revision,
        file=artifact.name,
        sha256=checksum,
        backend="memory" if artifact.suffix == ".json" else SQLITE_BACKEND,
    )
    target = staging / "manifest.json"
    target.write_text(json.dumps(manifest), encoding=TEXT_ENCODING)
    os.replace(target, folder / target.name)

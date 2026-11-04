"""Internal CorpusPersistence operations."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Callable, TypeVar, cast

from odr.corpus.index import HierarchyIndex
from odr.corpus.publication import open_published, publish_snapshot
from odr.schema import CorpusSnapshot
from odr.schema.arguments import StorageArguments
from odr.storage.tree import InMemoryTreeStorage

CorpusT = TypeVar("CorpusT", bound="CorpusPersistence")


class CorpusPersistence:
    """Snapshot import, export and revision publication operations.

    :ivar index: Index pinned to the built or opened corpus, or None before construction.
    :ivar storage_settings: Validated persistence settings.
    """

    index: HierarchyIndex | None
    storage_settings: StorageArguments

    @classmethod
    def from_snapshot(cls: type[CorpusT], snapshot: CorpusSnapshot) -> CorpusT:
        """Adopt an already validated corpus.

        :param snapshot: Validated immutable corpus snapshot.
        :returns: A facade or index pinned to the snapshot revision.
        """
        hierarchy = cast(Callable[..., CorpusT], cls)(chunker=None)
        hierarchy.index = HierarchyIndex.from_snapshot(snapshot)
        return hierarchy

    @classmethod
    def load_json(cls: type[CorpusT], path: str | Path) -> CorpusT:
        """Validate and load a portable snapshot.

        :param path: Source or destination filesystem path.
        :returns: A corpus loaded from the validated portable snapshot.
        """
        return cls.from_snapshot(
            CorpusSnapshot.model_validate_json(Path(path).read_text())
        )

    @classmethod
    async def open(cls: type[CorpusT], folder: str | Path) -> CorpusT:
        """Pin the published revision without loading source texts.

        :param folder: Published index directory.
        :returns: A corpus pinned to the currently published revision.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        hierarchy = cast(Callable[..., CorpusT], cls)(chunker=None)
        storage = await asyncio.to_thread(open_published, Path(folder))
        hierarchy.index = HierarchyIndex(storage)
        hierarchy.storage_settings = StorageArguments(
            storage_folder=str(folder),
            tree_backend="memory"
            if isinstance(storage, InMemoryTreeStorage)
            else "sqlite",
        )
        return hierarchy

    async def save_json(self, path: str | Path) -> None:
        """Explicitly materialize a portable snapshot.

        :param path: Source or destination filesystem path.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        storage = cast(HierarchyIndex, self.index).storage
        snapshot = await storage.export()
        await asyncio.to_thread(
            Path(path).write_text, snapshot.model_dump_json(), encoding="utf-8"
        )

    async def _publish(self, snapshot: CorpusSnapshot) -> None:
        """Publish an immutable revision and atomically replace its manifest.

        :param snapshot: Validated immutable corpus snapshot.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        storage = await asyncio.to_thread(
            publish_snapshot,
            Path(cast(str, self.storage_settings.storage_folder)),
            snapshot,
            self.storage_settings.tree_backend,
        )
        self.index = HierarchyIndex(storage)

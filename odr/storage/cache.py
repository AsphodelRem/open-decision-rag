"""Persistent JSON request cache and cancellation-safe singleflight."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import sqlite3
    from pathlib import Path
    from typing import Any
import asyncio
import json
import sqlite3
import time
from abc import ABC, abstractmethod
from contextlib import closing

SQLITE_TIMEOUT = 30


class CacheMissError(LookupError):
    """Replay requested an operation absent from the cache."""


class CacheCorruptionError(ValueError):
    """A cache entry cannot be decoded safely."""


class CacheBackend(ABC):
    """Versioned JSON cache contract."""

    async def __aenter__(self) -> CacheBackend:
        """Enter the component context.

        :returns: The opened object.
        """
        return self

    async def __aexit__(self, *exception: object) -> bool:
        """Exit the component context according to resource ownership.

        :param exception: Exception context or provider failure being handled.
        :returns: Whether to suppress the current exception.
        """
        return False

    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """Return a record or None.

        :param key: Semantic request cache key.
        :returns: Decoded unexpired cache entry, or no entry.
        """

    @abstractmethod
    async def put(self, key: str, response: Any, ttl: float | None = None) -> None:
        """Atomically store a successful response.

        :param key: Semantic request cache key.
        :param response: Successful provider response to validate or persist.
        :param ttl: Entry lifetime in seconds, or no expiration.
        """


class MemoryCache(CacheBackend):
    """In-process cache with the same serialization as SQLite."""

    def __init__(self) -> None:
        """Initialize an empty response cache."""
        self.records: dict[str, tuple[float, str]] = {}

    async def get(self, key: str) -> Any | None:
        """Return a copy of an unexpired response.

        :param key: Semantic request cache key.
        :returns: Decoded unexpired cache entry, or no entry.
        """
        record = self.records.get(key)
        if record is None or record[0] < time.time():
            return None
        return json.loads(record[1])

    async def put(self, key: str, response: Any, ttl: float | None = None) -> None:
        """Store a JSON response and expiration.

        :param key: Semantic request cache key.
        :param response: Successful provider response to validate or persist.
        :param ttl: Entry lifetime in seconds, or no expiration.
        """
        expires = float("inf") if ttl is None else time.time() + ttl
        self.records[key] = (expires, json.dumps(response, allow_nan=False))


class SQLiteCache(CacheBackend):
    """Durable request cache; each operation owns a short connection."""

    def __init__(self, path: str | Path) -> None:
        """Create or open the durable request cache.

        :param path: Source or destination filesystem path.
        """
        self.path = str(path)
        with closing(self._connect()) as connection, connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, expires REAL, response TEXT)"
            )

    async def get(self, key: str) -> Any | None:
        """Read off the event loop and reject corrupt JSON.

        :param key: Semantic request cache key.
        :returns: Decoded unexpired cache entry, or no entry.
        """
        return await asyncio.to_thread(self._read, key)

    async def put(self, key: str, response: Any, ttl: float | None = None) -> None:
        """Commit a complete response atomically.

        :param key: Semantic request cache key.
        :param response: Successful provider response to validate or persist.
        :param ttl: Entry lifetime in seconds, or no expiration.
        """
        encoded = json.dumps(response, allow_nan=False)
        expires = None if ttl is None else time.time() + ttl
        await asyncio.to_thread(self._write, key, encoded, expires)

    def _connect(self) -> sqlite3.Connection:
        """Open a short-lived SQLite cache connection.

        :returns: A SQLite connection owned by the caller.
        """
        return sqlite3.connect(self.path, timeout=SQLITE_TIMEOUT)

    def _read(self, key: str) -> Any | None:
        """Read and decode one unexpired JSON cache entry.

        :param key: Semantic request cache key.
        :returns: Decoded unexpired cache entry, or no entry.
        :raises CacheCorruptionError: If the input, stored data, or operation result is invalid.
        """
        with closing(self._connect()) as connection, connection:
            record = connection.execute(
                "SELECT expires,response FROM cache WHERE key=?", (key,)
            ).fetchone()
        if record is None:
            return None
        if record[0] is not None and record[0] < time.time():
            return None
        try:
            return json.loads(record[1])
        except (ValueError, TypeError) as exception:
            raise CacheCorruptionError(key) from exception

    def _write(self, key: str, encoded: str, expires: float | None) -> None:
        """Commit complete records in a short SQLite transaction.

        :param key: Semantic request cache key.
        :param encoded: Serialized JSON response.
        :param expires: Expiration timestamp in seconds, or no expiration.
        """
        with closing(self._connect()) as connection, connection:
            connection.execute(
                "INSERT OR REPLACE INTO cache VALUES (?,?,?)", (key, expires, encoded)
            )

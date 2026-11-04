"""Explicit read-only SQLite connections for published corpus artifacts."""

import sqlite3
from pathlib import Path


def read_connection(path: str | Path) -> sqlite3.Connection:
    """Open an existing index without creating or modifying its database.

    :param path: Published SQLite artifact.
    :returns: Connection that must be explicitly closed by its owner.
    """
    uri = Path(path).resolve().as_uri()
    return sqlite3.connect(f"{uri}?mode=ro", uri=True)

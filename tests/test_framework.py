"""Offline integration checks for indexing, retrieval and persistent cache."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import httpx
import asyncio
import tempfile
import unittest
from pathlib import Path

import httpx

from odr import (
    CachedAsyncOpenAI,
    CorpusHierarchy,
    DecisionSearchEngine,
    Document,
    MockDecisionModel,
    SQLiteCache,
    StorageArguments,
)

SUCCESS_STATUS = 200
INSTALL_ID = "install"


class FrameworkTests(unittest.TestCase):
    """End-to-end operations without model services."""

    def test_restart(self) -> None:
        """Search remains identical after reopening SQLite."""
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(self._restart(folder))

    def test_cache_replay(self) -> None:
        """A new client replays full provider JSON without transport."""
        with tempfile.TemporaryDirectory() as folder:
            asyncio.run(self._replay(Path(folder) / "cache.sqlite3"))

    async def _restart(self, folder: str | Path) -> None:
        """Verify that persisted and reopened indexes produce identical retrieval.

        :param folder: Published index directory.
        """
        hierarchy = CorpusHierarchy(
            storage_settings=StorageArguments(storage_folder=folder)
        )
        await hierarchy.build_from_docs(
            [
                Document(id=INSTALL_ID, title="Install", text="Install the package."),
                Document(
                    id="billing", title="Billing", text="Invoices arrive monthly."
                ),
            ]
        )
        assert hierarchy.index is not None
        first = await DecisionSearchEngine(hierarchy, MockDecisionModel()).search(
            INSTALL_ID
        )
        reopened = await CorpusHierarchy.open(folder)
        assert reopened.index is not None
        second = await DecisionSearchEngine(reopened, MockDecisionModel()).search(
            INSTALL_ID
        )
        self.assertEqual(first, second)
        self.assertEqual(first.passages[0].document_id, INSTALL_ID)
        exported = Path(folder) / "snapshot.json"
        await reopened.save_json(exported)
        hierarchy = CorpusHierarchy.load_json(exported)
        assert hierarchy.index is not None
        self.assertEqual(hierarchy.index.revision, reopened.index.revision)

    async def _replay(self, path: str | Path) -> None:
        """Verify persistent replay and in-process request singleflight.

        :param path: Source or destination filesystem path.
        """
        calls: list[httpx.Request] = []
        transport = httpx.AsyncClient(transport=httpx.MockTransport(Responder(calls)))
        async with CachedAsyncOpenAI(
            "http://localhost/v1", SQLiteCache(path), http_client=transport
        ) as client:
            responses = await asyncio.gather(
                *(
                    client.request("chat/completions", {"model": "local"})
                    for index in range(8)
                )
            )
            self.assertEqual(len(calls), 1)
            self.assertEqual(responses[0], responses[-1])
        async with CachedAsyncOpenAI(
            "http://localhost/v1", SQLiteCache(path), cache_mode="replay"
        ) as replay:
            self.assertEqual(
                await replay.request("chat/completions", {"model": "local"}),
                responses[0],
            )
            self.assertEqual(replay.stats["network_attempts"], 0)
        await transport.aclose()


if __name__ == "__main__":
    unittest.main()


class Responder:
    """Track requests made through an injected transport."""

    def __init__(self, calls: list[httpx.Request]) -> None:
        """Bind the deterministic HTTP transport to its request log.

        :param calls: Mutable request log owned by the test.
        """
        self.calls = calls

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Record the test request and return a deterministic provider response.

        :param request: HTTP request handled by the deterministic test transport.
        :returns: A configured component or deterministic HTTP response.
        """
        self.calls.append(request)
        return httpx.Response(
            SUCCESS_STATUS, json={"choices": [], "usage": {"total_tokens": 7}}
        )

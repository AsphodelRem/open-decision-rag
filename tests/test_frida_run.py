"""Exercise the complete FRIDA experiment script without external model services."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

import odr
from examples import frida_decisions as experiment

RELEVANCE = 0.8
SUCCESS = 200


class ModelServer:
    """Serve deterministic embedding, summary, answer and native pooling fixtures."""

    def __init__(self) -> None:
        """Initialize physical endpoint accounting."""
        self.paths: list[str] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        """Return independently constructed wire responses for each real adapter.

        :param request: Physical request from the shared execution layer.
        :returns: Provider response for embedding, chat or FRIDA pooling.
        """
        self.paths.append(request.url.path)
        payload = json.loads(request.content)
        if request.url.path.endswith("embeddings"):
            return httpx.Response(
                SUCCESS,
                json=dict(
                    data=[
                        dict(index=index, embedding=[index + 1, 1])
                        for index, text in enumerate(payload["input"])
                    ]
                ),
            )
        if request.url.path == "/pooling":
            return httpx.Response(
                SUCCESS,
                json=dict(
                    data=dict(
                        answers={
                            identity: self.answer(question)
                            for identity, question in payload["data"][
                                "questions"
                            ].items()
                        },
                        usage=dict(state_truncated=False),
                    )
                ),
            )
        return httpx.Response(
            SUCCESS,
            json=dict(
                choices=[dict(message=dict(content="A concise factual summary."))]
            ),
        )

    def answer(self, question: dict[str, object]) -> dict[str, object]:
        """Supply native probabilities without exposing expected source labels.

        :param question: Parsed native question definition.
        :returns: Independent binary score or a uniform choice distribution.
        """
        if question["type"] == "noul":
            return dict(type="noul", noul=RELEVANCE)
        criteria = question["criteria"]
        if not isinstance(criteria, dict):
            raise ValueError("Expected choice criteria")
        return dict(
            type="choice",
            choice=next(iter(criteria)),
            probabilities={identity: 1 / len(criteria) for identity in criteria},
        )


class FridaRunTests(unittest.IsolatedAsyncioTestCase):
    """Full build, publication, answer and persistent cache-only rebuild checks."""

    async def asyncSetUp(self) -> None:
        """Own a shared fixture transport and scoped environment settings."""
        self.server = ModelServer()
        self.transport = httpx.AsyncClient(transport=httpx.MockTransport(self.server))
        self.addAsyncCleanup(self.transport.aclose)

    def client(self, role: str, cache: odr.SQLiteCache) -> odr.CachedAsyncOpenAI:
        """Inject a controlled transport into otherwise unchanged HTTP adapters.

        :param role: Build or decision deployment role.
        :param cache: Real persistent cache owned by the experiment.
        :returns: Shared-client configuration for this test deployment.
        """
        return odr.CachedAsyncOpenAI(
            "http://localhost/v1" if role == "BUILD" else "http://localhost",
            cache=cache,
            http_client=self.transport,
        )

    async def test_complete_run_and_offline_rebuild(self) -> None:
        """A fresh experiment rebuilds and answers from cache with no new requests."""
        with tempfile.TemporaryDirectory() as folder:
            settings = self.configuration(Path(folder))
            with (
                patch.dict(
                    os.environ,
                    dict(
                        EMBEDDING_MODEL="embedding",
                        CHAT_MODEL="chat",
                        DECISION_MODEL="frida",
                        DEPLOYMENT_REVISION="fixture-v1",
                    ),
                ),
                patch.object(experiment.real_models, "configured_client", self.client),
                patch.object(experiment.sys.stdout, "write"),
            ):
                report = await experiment.FridaRun(
                    settings,
                    experiment.RunOptions(
                        reuse_index=False,
                        cache_mode="read_write",
                    ),
                ).run()
                self.assertEqual(
                    report["checks"],
                    dict(sqlite_reopen=True, json_roundtrip=True, replay_equal=True),
                )
                self.assertIn("/v1/embeddings", self.server.paths)
                self.assertIn("/pooling", self.server.paths)
                self.server.paths.clear()
                await experiment.FridaRun(
                    settings,
                    experiment.RunOptions(
                        reuse_index=False,
                        cache_mode="replay",
                    ),
                ).run()
                self.assertFalse(self.server.paths)
            self.assertTrue((settings.output / "result.json").is_file())
            self.assertTrue((settings.output / "snapshot.json").is_file())

    async def test_chat_prepares_once(self) -> None:
        """Two interactive questions reuse the same built index and retain history."""
        with tempfile.TemporaryDirectory() as folder:
            runner = experiment.FridaRun(
                self.configuration(Path(folder)),
                experiment.RunOptions(
                    reuse_index=False, cache_mode="read_write", chat=True
                ),
            )
            with (
                patch.dict(
                    os.environ,
                    dict(
                        EMBEDDING_MODEL="embedding",
                        CHAT_MODEL="chat",
                        DECISION_MODEL="frida",
                        DEPLOYMENT_REVISION="fixture-v1",
                    ),
                ),
                patch.object(experiment.real_models, "configured_client", self.client),
                patch.object(
                    experiment.frida_chat,
                    "read_question",
                    side_effect=[
                        "How to install?",
                        "And configuration?",
                        "/exit",
                    ],
                ),
                patch.object(experiment.frida_chat, "display_answer"),
                patch.object(experiment.sys.stdout, "write"),
                patch.object(
                    runner, "prepare", AsyncMock(wraps=runner.prepare)
                ) as prepare,
            ):
                await runner.run()
                prepare.assert_awaited_once()
            transcript = (
                (runner.settings.output / "chat.jsonl").read_text().splitlines()
            )
            self.assertEqual(len(transcript), 2)
            history = json.loads(transcript[1])["response"]["query"]["messages"]
            self.assertEqual(history[0], ["user", "How to install?"])
            self.assertEqual(self.server.paths.count("/v1/embeddings"), 1)

    def configuration(self, folder: Path) -> experiment.RunSettings:
        """Create a small real sklearn build with deterministic model fixtures.

        :param folder: Isolated experiment workspace.
        :returns: Typed build and search configuration.
        """
        corpus = folder / "sources"
        corpus.mkdir()
        for index in range(3):
            (corpus / f"{index}.txt").write_text(
                f"Document {index}. Installation instructions."
            )
        return experiment.RunSettings(
            corpus=corpus,
            output=folder / "experiment",
            builder=odr.BuilderArguments(hierarchy_type="kmeans", n_clusters=2),
        )

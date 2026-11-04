"""Console commands, bounded conversation history and recoverable model errors."""

import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import httpx

import odr
from examples import frida_chat


def answer(query: odr.SearchQuery) -> odr.SearchEngineResponse:
    """Create a typed successful answer without inference dependencies.

    :param query: Original query with the history supplied by the chat.
    :returns: Deterministic answer and source-identified evidence.
    """
    return odr.SearchEngineResponse(
        query=query,
        response="An answer.",
        retrieval=odr.DecisionSearchRetrieve(
            query=query,
            corpus_revision="revision",
            config_fingerprint="config",
            passages=(
                odr.RetrievedPassage(
                    node_id="passage",
                    chunk_id="chunk",
                    document_id="document",
                    text="Source evidence.",
                    path=("root", "document", "passage"),
                    rank=1,
                    binary_score=1,
                    ranking_score=1,
                ),
            ),
        ),
    )


class FridaChatTests(unittest.IsolatedAsyncioTestCase):
    """One engine serves successive questions with explicit conversation context."""

    async def test_commands_and_history(self) -> None:
        """Help, blank lines and clear do not query the engine; sources are displayed."""
        engine = Mock(spec=odr.DecisionSearchEngine)
        engine.query = AsyncMock(side_effect=answer)
        with tempfile.TemporaryDirectory() as folder:
            chat = frida_chat.FridaChat(engine, Path(folder) / "chat.jsonl")
            with (
                patch.object(
                    frida_chat,
                    "read_question",
                    side_effect=[
                        "",
                        "/help",
                        "first",
                        "follow-up",
                        "/clear",
                        "new topic",
                        "/exit",
                    ],
                ),
                patch.object(
                    frida_chat.sys, "stdout", new_callable=io.StringIO
                ) as output,
            ):
                await chat.run()
                self.assertIn("chunk", output.getvalue())
                self.assertIn("Source evidence.", output.getvalue())
            self.assertEqual(engine.query.await_count, 3)
            queries = [call.args[0] for call in engine.query.call_args_list]
            self.assertEqual(
                queries[1].messages,
                (
                    ("user", "first"),
                    ("assistant", "An answer."),
                ),
            )
            self.assertEqual(queries[2].messages, ())
            self.assertEqual(len(chat.log_path.read_text().splitlines()), 3)

    async def test_errors_do_not_commit_history(self) -> None:
        """Model, replay and transport failures allow the next question to succeed."""
        engine = Mock(spec=odr.DecisionSearchEngine)
        engine.query = AsyncMock(
            side_effect=[
                ValueError("context too large"),
                odr.CacheMissError("missing"),
                httpx.ConnectError("offline"),
                answer(odr.SearchQuery(text="success")),
            ]
        )
        with tempfile.TemporaryDirectory() as folder:
            chat = frida_chat.FridaChat(engine, Path(folder) / "chat.jsonl")
            with (
                patch.object(
                    frida_chat,
                    "read_question",
                    side_effect=[
                        "first",
                        "second",
                        "third",
                        "success",
                        "/quit",
                    ],
                ),
                patch.object(frida_chat.sys, "stdout", new_callable=io.StringIO),
                patch.object(frida_chat.sys, "stderr", new_callable=io.StringIO),
            ):
                await chat.run()
            self.assertEqual(
                chat.history, [("user", "success"), ("assistant", "An answer.")]
            )
            self.assertEqual(len(chat.log_path.read_text().splitlines()), 1)

    async def test_history_window_and_log(self) -> None:
        """Keep six recent messages and append complete response/diagnostic records."""
        engine = Mock(spec=odr.DecisionSearchEngine)
        engine.query = AsyncMock(side_effect=answer)
        with tempfile.TemporaryDirectory() as folder:
            chat = frida_chat.FridaChat(engine, Path(folder) / "chat.jsonl")
            with patch.object(frida_chat, "display_answer"):
                index = 0
                while index < 4:
                    await chat.ask(str(index))
                    index += 1
            self.assertEqual(len(chat.history), frida_chat.HISTORY_MESSAGES)
            self.assertEqual(chat.history[0], ("user", "1"))
            record = json.loads(chat.log_path.read_text().splitlines()[-1])
            self.assertEqual(record["response"]["query"]["text"], "3")

    def test_end_of_input(self) -> None:
        """EOF maps to a clean exit without an infinite blank-line loop."""
        with (
            patch.object(frida_chat.sys, "stdin", io.StringIO("")),
            patch.object(frida_chat.sys, "stdout", io.StringIO()),
        ):
            self.assertEqual(frida_chat.read_question(), "/exit")

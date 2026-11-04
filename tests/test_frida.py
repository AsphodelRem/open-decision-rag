"""Independent pooling fixtures for FRIDA request, readout and replay contracts."""

import asyncio
import json
import tempfile
from pathlib import Path

import test_decision_models as fixtures

from odr import (
    CachedAsyncOpenAI,
    CorpusHierarchy,
    DecisionModelFRIDA,
    DecisionSearchEngine,
)
from odr.models.frida import frida_request, frida_scores
from odr.models.types import WireResponse
from odr.schema import Document, SearchArguments
from odr.storage.cache import CacheMissError, SQLiteCache

MODEL = "ai-forever/FRIDA-Decisions"
DEPLOYMENT = "frida-fixture-v1"
YES_RELEVANT = 0.8
YES_OTHER = 0.7
CHOICE_OTHER = 0.2
LOW_CONFIDENCE = 0.01
PROMPT_TOKENS = 120
OPTION_IDS = ("c0", "c1")


def binary_response() -> WireResponse:
    """Build a native pooling response with independent, non-normalized scores.

    :returns: Full wire response including both provider usage envelopes.
    """
    return dict(
        data=dict(
            answers=dict(
                c1=dict(type="noul", noul=YES_OTHER),
                c0=dict(type="noul", noul=YES_RELEVANT),
            ),
            usage=dict(state_truncated=False, state_tokens=100),
        ),
        usage=dict(prompt_tokens=PROMPT_TOKENS),
    )


def choice_response() -> WireResponse:
    """Keep native option probabilities distinct from confidence and raw margins.

    :returns: Full choice wire response in neutral option identity space.
    """
    return dict(
        data=dict(
            answers=dict(
                selection=dict(
                    type="choice",
                    choice=OPTION_IDS[1],
                    probabilities=dict(c0=CHOICE_OTHER, c1=YES_RELEVANT),
                    confidence=LOW_CONFIDENCE,
                )
            )
        )
    )


def provider_response(response: WireResponse) -> WireResponse:
    """Unwrap the native pooling result for fixture inspection.

    :param response: Full pooling wire response.
    :returns: Native FRIDA response dictionary.
    """
    return response["data"]


class FridaCase(fixtures.DecisionTestCase):
    """Reuse the controlled transport with a root pooling endpoint."""

    async def asyncSetUp(self) -> None:
        """Own a root-URL client while retaining the fixture transport lifecycle."""
        await super().asyncSetUp()
        self.pooling = CachedAsyncOpenAI(
            "http://localhost", http_client=self.client._transport, api_key="local"
        )
        self.addAsyncCleanup(self.pooling.__aexit__)

    def model(self, revision: str = DEPLOYMENT) -> DecisionModelFRIDA:
        """Create a fixture model with explicit deployment provenance.

        :param revision: Weights and server configuration identity.
        :returns: Configured native pooling adapter.
        """
        return DecisionModelFRIDA(self.pooling, MODEL, deployment_revision=revision)


class FridaWireTests(FridaCase):
    """Wire shape, shared context and degenerate input behavior."""

    async def test_binary_wire(self) -> None:
        """One request retains source order, query and history with neutral IDs."""
        self.queue.responses.append(binary_response())
        self.assertEqual(
            await self.model().binary_judge(self.state, self.candidates),
            dict(install=YES_RELEVANT, billing=YES_OTHER),
        )
        request = self.queue.requests[0]
        payload = json.loads(request.content)
        self.assertEqual(request.url.path, "/pooling")
        self.assertEqual(request.headers["Authorization"], "Bearer local")
        self.assertEqual(
            provider_response(payload)["state"]["query"], self.state["query"]
        )
        history = provider_response(payload)["state"]["history"]
        self.assertEqual(history, [["user", "I use Linux"]])
        self.assertEqual(
            provider_response(payload)["state"]["candidates"],
            [
                dict(id="c0", text="Install the package"),
                dict(id="c1", text="Monthly invoices"),
            ],
        )
        self.assertEqual(set(provider_response(payload)["questions"]), set(OPTION_IDS))

    async def test_choice(self) -> None:
        """Read probabilities and restore original IDs without using confidence."""
        self.queue.responses.append(choice_response())
        self.assertEqual(
            await self.model().choice_judge(self.state, self.candidates),
            dict(install=CHOICE_OTHER, billing=YES_RELEVANT),
        )
        payload = json.loads(self.queue.requests[0].content)
        self.assertEqual(
            set(provider_response(payload)["questions"]["selection"]["criteria"]),
            set(OPTION_IDS),
        )

    async def test_degenerate_inputs(self) -> None:
        """Empty batches and a certain single choice never require HTTP."""
        self.assertEqual(await self.model().binary_judge(self.state, {}), {})
        self.assertEqual(await self.model().choice_judge(self.state, {}), {})
        self.assertEqual(
            await self.model().choice_judge(self.state, dict(only="passage")),
            dict(only=1.0),
        )
        self.assertFalse(self.queue.requests)

    def test_context_renderer(self) -> None:
        """Character budgets use the same native request renderer as dispatch."""
        for choice in (False, True):
            expected = json.dumps(
                frida_request(self.state, self.candidates, choice), ensure_ascii=False
            )
            self.assertEqual(
                self.model().context_size(self.state, self.candidates, choice),
                len(expected),
            )


class FridaValidationTests(FridaCase):
    """Reject invalid evidence before caching and preserve replay semantics."""

    async def test_truncated_not_cached(self) -> None:
        """Truncated shared context is rejected and a corrected response can replace it."""
        truncated = binary_response()
        provider_response(truncated)["usage"]["state_truncated"] = True
        self.queue.responses.extend([truncated, binary_response()])
        with self.assertRaisesRegex(ValueError, "truncated"):
            await self.model().binary_judge(self.state, self.candidates)
        self.assertEqual(
            await self.model().binary_judge(self.state, self.candidates),
            dict(install=YES_RELEVANT, billing=YES_OTHER),
        )
        self.assertEqual(len(self.queue.requests), 2)

    def test_invalid_binary(self) -> None:
        """Envelopes, IDs, discriminators and strict probabilities must be valid."""
        invalid: list[WireResponse] = [{}, dict(data=[])]
        invalid.append(dict(data=dict(answers={})))
        for score in (True, "0.5", float("nan"), float("inf"), -0.1, 1.1):
            response = binary_response()
            provider_response(response)["answers"][OPTION_IDS[0]]["noul"] = score
            invalid.append(response)
        for response in invalid:
            with self.subTest(response=response):
                with self.assertRaises(ValueError):
                    frida_scores(response, self.candidates)

    def test_invalid_choice(self) -> None:
        """Unknown options, selected IDs and unnormalized distributions fail."""
        for update in (
            dict(choice="unknown"),
            dict(probabilities=dict(c0=0.5, c1=YES_RELEVANT)),
            dict(probabilities=dict(c0=1.0)),
            dict(type="ranking"),
        ):
            response = choice_response()
            provider_response(response)["answers"]["selection"].update(update)
            with self.assertRaises(ValueError):
                frida_scores(response, self.candidates, choice=True)

    async def test_singleflight_and_revision(self) -> None:
        """Concurrent identical calls share one request; deployments remain isolated."""
        self.queue.responses.extend([binary_response(), binary_response()])
        await asyncio.gather(
            *(
                self.model().binary_judge(self.state, self.candidates)
                for index in range(3)
            )
        )
        self.assertEqual(len(self.queue.requests), 1)
        await self.model("another-deployment").binary_judge(self.state, self.candidates)
        self.assertEqual(len(self.queue.requests), 2)

    async def test_persistent_replay(self) -> None:
        """A fresh client replays SQLite wire responses and fails offline on misses."""
        with tempfile.TemporaryDirectory() as folder:
            self.pooling.cache = SQLiteCache(Path(folder) / "cache.sqlite3")
            self.queue.responses.append(choice_response())
            expected = await self.model().choice_judge(self.state, self.candidates)
            async with CachedAsyncOpenAI(
                "http://localhost",
                cache=SQLiteCache(Path(folder) / "cache.sqlite3"),
                cache_mode="replay",
                http_client=self.client._transport,
            ) as fresh:
                model = DecisionModelFRIDA(fresh, MODEL, deployment_revision=DEPLOYMENT)
                self.assertEqual(
                    await model.choice_judge(self.state, self.candidates), expected
                )
                with self.assertRaises(CacheMissError):
                    await model.binary_judge(self.state, self.candidates)
            self.assertEqual(len(self.queue.requests), 1)

    async def test_search_integration(self) -> None:
        """Native binary and choice responses drive the real traversal and ranking."""
        corpus = await CorpusHierarchy().build_from_docs(
            [
                Document(id="install", title="Install", text="Install the package"),
                Document(id="billing", title="Billing", text="Monthly invoices"),
            ]
        )
        self.queue.responses.extend(
            [binary_response(), binary_response(), choice_response()]
        )
        engine = DecisionSearchEngine(
            corpus, self.model(), search_settings=SearchArguments(choice_top_m=2)
        )
        retrieval = await engine.search("installation")
        self.assertEqual(retrieval.passages[0].document_id, "billing")
        self.assertEqual(len(self.queue.requests), 3)

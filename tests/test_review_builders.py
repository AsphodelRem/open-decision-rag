"""Regression checks for deterministic summaries and actual component provenance."""

from __future__ import annotations

import asyncio
import unittest
from collections.abc import Sequence
from typing import Any

from odr import (
    BuilderArguments,
    CorpusHierarchy,
    DecisionSearchEngine,
    KMeansHierarchyBuilder,
    LLMDescriptionBuilder,
    MetadataHierarchyBuilder,
    MockDecisionModel,
    MockEmbedder,
)
from odr.models.mock import MockLLM
from odr.models.types import Messages, Vectors
from odr.schema import Document

SUMMARY_DELAY = 0.01


class DelayedSummary(MockLLM):
    """Vary transport timing without changing semantic generation settings."""

    slow_title = "A"

    async def chat_completion(self, messages: Messages) -> str:
        """Delay one deterministic summary to invert completion order.

        :param messages: Summary request.
        :returns: Unchanged deterministic summary text.
        """
        if messages[-1]["content"] == self.slow_title:
            await asyncio.sleep(SUMMARY_DELAY)
        return await super().chat_completion(messages)


class FixedPartition(KMeansHierarchyBuilder):
    """Isolate asynchronous summary scheduling from numerical clustering."""

    def _partition(self, members: Sequence[int], vectors: Vectors) -> list[list[int]]:
        """Return a fixed valid partition.

        :param members: Original document row indices.
        :param vectors: Unused embedding features.
        :returns: Fixed singleton groups.
        """
        return [[0], [1]]


class VersionedJudge(MockDecisionModel):
    """Expose an explicit semantic revision for custom decision implementations."""

    def __init__(self, revision: str) -> None:
        """Bind a semantic revision without changing fixture relevance scores.

        :param revision: User-owned decision implementation revision.
        """
        self.revision = revision

    def semantic_identity(self) -> object:
        """Declare stable user component provenance.

        :returns: Explicit semantic revision.
        """
        return dict(revision=self.revision)


class BuilderRegressionTests(unittest.IsolatedAsyncioTestCase):
    """Actual injected components determine reproducible corpus identities."""

    async def test_summary_completion_order(self) -> None:
        """Different summary response schedules produce identical snapshots."""
        first, second = await asyncio.gather(
            self.delayed_build("A"), self.delayed_build("B")
        )
        assert first.index is not None and second.index is not None
        self.assertEqual(first.index.revision, second.index.revision)
        self.assertEqual(
            await first.index.storage.export(), await second.index.storage.export()
        )

    async def delayed_build(self, slow: str) -> CorpusHierarchy:
        """Build the same fixed tree with one intentionally delayed summary.

        :param slow: Title whose summary responds later.
        :returns: Completed corpus.
        """
        llm = DelayedSummary()
        llm.slow_title = slow
        builder = FixedPartition(
            MockEmbedder(), LLMDescriptionBuilder(llm), BuilderArguments(n_clusters=2)
        )
        corpus = CorpusHierarchy(builder=builder)
        await corpus.build_from_docs(
            [Document(id=title, title=title, text=title) for title in ("A", "B")]
        )
        return corpus

    async def test_actual_builder_fingerprint(self) -> None:
        """Explicit builder overrides are reflected in build provenance."""
        source = [
            Document(id="manual", title="Manual", text="Install", group_path=("Help",))
        ]
        flat = await CorpusHierarchy().build_from_docs(source)
        grouped = await CorpusHierarchy(
            builder=MetadataHierarchyBuilder()
        ).build_from_docs(source)
        assert flat.index is not None and grouped.index is not None
        first = await flat.index.storage.export()
        second = await grouped.index.storage.export()
        self.assertNotEqual(first.build_fingerprint, second.build_fingerprint)
        self.assertTrue(first.manifest)

    async def test_judge_fingerprint(self) -> None:
        """Different decision implementation revisions have different run identities."""
        corpus = await CorpusHierarchy().build_from_docs(
            [Document(id="manual", text="Install")]
        )
        first = await DecisionSearchEngine(corpus, VersionedJudge("v1")).search(
            "install"
        )
        second = await DecisionSearchEngine(corpus, VersionedJudge("v2")).search(
            "install"
        )
        self.assertNotEqual(first.config_fingerprint, second.config_fingerprint)
        self.assertEqual(first.passages, second.passages)

    async def test_unsupported_configuration(self) -> None:
        """Unsupported options are rejected instead of silently ignored."""
        with self.assertRaises(ValueError):
            BuilderArguments.model_validate(dict(make_group_summary=False))
        with self.assertRaises(NotImplementedError):
            await CorpusHierarchy().build_from_docs(
                [Document(id="manual", text="Install")], resume=True
            )
        invalid: dict[str, Any] = dict(llmm=MockLLM())
        with self.assertRaises(TypeError):
            CorpusHierarchy(**invalid)

    async def test_recursive_branching(self) -> None:
        """Recursive child splitting uses branching_factor rather than n_clusters."""
        corpus = CorpusHierarchy(
            embedder=MockEmbedder(),
            llm=MockLLM(),
            builder_settings=BuilderArguments(
                hierarchy_type="recursive",
                n_clusters=1,
                branching_factor=2,
                max_documents_per_group=1,
                max_depth=2,
            ),
        )
        await corpus.build_from_docs(
            [
                Document(id=str(index), title=str(index), text=f"Text {index}")
                for index in range(3)
            ]
        )
        assert corpus.index is not None
        group = (await corpus.index.children(corpus.index.root_id))[0]
        self.assertEqual(len(await corpus.index.children(group.id)), 2)

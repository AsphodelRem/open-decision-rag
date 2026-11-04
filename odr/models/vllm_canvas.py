"""Compile one-question DiffusionGemma canvases using a pinned tokenizer."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from importlib import import_module
from typing import Callable, cast

from odr.models.client import CachedAsyncOpenAI
from odr.models.protocols import DecisionTokenizer
from odr.models.types import CacheMode, Messages
from odr.models.vllm_types import CanvasPlan, PromptCount
from odr.schema import fingerprint
from odr.schema.arguments import VLLMDecisionArguments
from odr.storage.cache import CacheMissError

ADAPTER_REVISION = "diffusiongemma-vllm-v2"
CANVAS_STEP = 16
THOUGHT_SCAFFOLD = "<|channel>thought\n<channel|>"


def answer_position(variants: list[list[int]]) -> int:
    """Prove that every label changes exactly one shared template token.

    :param variants: Complete template tokenizations for each allowed label.
    :returns: Sole answer-slot position.
    :raises ValueError: If labels change token count or more than one position.
    """
    if len({len(variant) for variant in variants}) != 1:
        raise ValueError("Decision labels change the template token count")
    positions = {
        index
        for variant in variants[1:]
        for index, token in enumerate(variant)
        if token != variants[0][index]
    }
    if len(positions) != 1:
        raise ValueError("Decision labels must occupy one shared token position")
    return next(iter(positions))


class CanvasCompiler:
    """Lazily load tokenizer metadata without loading model weights."""

    def __init__(
        self,
        settings: VLLMDecisionArguments,
        tokenizer: DecisionTokenizer | None = None,
    ) -> None:
        """Configure tokenizer identity and an optional injected implementation.

        :param settings: Pinned deployment and tokenizer configuration.
        :param tokenizer: Explicit compatible tokenizer, primarily for offline use.
        """
        self.settings = settings
        self.tokenizer = tokenizer

    def compile(self, labels: tuple[str, ...]) -> CanvasPlan:
        """Resolve the answer slot and construct a padded diffusion template.

        :param labels: Ordered answer labels, such as yes/no or A/B/C.
        :returns: Validated tokenizer-specific canvas plan.
        :raises ValueError: If template tokens or labels are incompatible.
        """
        tokenizer = self._load()
        self._special("<|channel>")
        self._special("<channel|>")
        variants = [
            tokenizer.encode(THOUGHT_SCAFFOLD, add_special_tokens=False)
            + tokenizer.encode(f"q1: {label}", add_special_tokens=False)
            for label in labels
        ]
        position = answer_position(variants)
        canvas = variants[0] + [self._special("<turn|>")]
        width = -(-len(canvas) // CANVAS_STEP) * CANVAS_STEP
        if width > self.settings.canvas_length:
            raise ValueError("Decision template exceeds the configured canvas")
        return CanvasPlan(
            canvas=canvas
            + [self._special("<pad>") for index in range(width - len(canvas))],
            position=position,
            output_length=len(canvas),
            label_ids=[variant[position] for variant in variants],
            labels=list(labels),
            vocabulary_size=len(tokenizer),
        )

    def token_count(self, messages: Messages) -> int:
        """Apply the matching server chat template before measuring input tokens.

        :param messages: Exact model instruction and shared-context messages.
        :returns: Complete input token count including generation prompt markers.
        """
        encoded = self._load().apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        return len(encoded["input_ids"] if isinstance(encoded, Mapping) else encoded)

    def _special(self, spelling: str) -> int:
        """Resolve a required boundary marker without assuming token IDs.

        :param spelling: Exact special-token spelling from the readout profile.
        :returns: Its unique tokenizer ID.
        :raises ValueError: If the tokenizer does not recognize a single token.
        """
        tokens = self._load().encode(spelling, add_special_tokens=False)
        if (
            len(tokens) != 1
            or self._load().decode(tokens, skip_special_tokens=False) != spelling
        ):
            raise ValueError(f"Expected a single special token: {spelling}")
        return tokens[0]

    def _load(self) -> DecisionTokenizer:
        """Load only the configured tokenizer when compilation is necessary.

        :returns: Loaded or explicitly injected tokenizer.
        """
        if self.tokenizer is None:
            factory = cast(
                Callable[..., DecisionTokenizer],
                import_module("transformers").AutoTokenizer.from_pretrained,
            )
            self.tokenizer = factory(
                self.settings.tokenizer_name,
                revision=self.settings.tokenizer_revision,
                trust_remote_code=False,
            )
        return self.tokenizer


class TokenizerCache:
    """Persist compiled plans and prompt counts independently of loaded tokenizers."""

    def __init__(
        self,
        client: CachedAsyncOpenAI,
        settings: VLLMDecisionArguments,
        tokenizer: DecisionTokenizer | None,
    ) -> None:
        """Bind a shared cache and lazy tokenizer compiler.

        :param client: Shared HTTP execution and cache policy.
        :param settings: Pinned tokenizer and deployment configuration.
        :param tokenizer: Optional explicitly injected tokenizer.
        """
        self.client = client
        self.compiler = CanvasCompiler(settings, tokenizer)
        self.lock = asyncio.Lock()

    async def plan(self, labels: tuple[str, ...]) -> CanvasPlan:
        """Replay persisted tokenizer plans without loading or downloading a tokenizer.

        :param labels: Ordered allowed answer labels.
        :returns: Validated cached or newly compiled tokenizer plan.
        :raises CacheMissError: If replay cannot find the required plan.
        """
        key = fingerprint(
            [ADAPTER_REVISION, self.compiler.settings.model_dump(), labels]
        )
        async with self.lock:
            raw = None
            if self.client.cache_mode not in {CacheMode.DISABLED, CacheMode.REFRESH}:
                raw = await self.client.cache.get(key)
            if raw is not None:
                plan = CanvasPlan.model_validate(raw, strict=True)
                if (
                    plan.labels != list(labels)
                    or len(plan.canvas) > self.compiler.settings.canvas_length
                ):
                    raise ValueError(
                        "Cached vLLM plan does not match its readout configuration"
                    )
                return plan
            if self.client.cache_mode == "replay":
                raise CacheMissError(f"Missing vLLM tokenizer plan: {key}")
            plan = await asyncio.to_thread(self.compiler.compile, labels)
            if self.client.cache_mode != CacheMode.DISABLED:
                await self.client.cache.put(key, plan.model_dump())
            return plan

    async def count(self, messages: Messages) -> int:
        """Replay cached exact prompt counts without tokenizer loading.

        :param messages: Fully rendered model input.
        :returns: Positive input token count.
        :raises CacheMissError: If replay has no count for this exact prompt.
        """
        key = fingerprint(
            [
                ADAPTER_REVISION,
                "token-count",
                self.compiler.settings.model_dump(),
                messages,
            ]
        )
        async with self.lock:
            raw = (
                None
                if self.client.cache_mode in {CacheMode.DISABLED, CacheMode.REFRESH}
                else await self.client.cache.get(key)
            )
            if raw is not None:
                return PromptCount(count=raw).count
            if self.client.cache_mode == "replay":
                raise CacheMissError(f"Missing tokenizer count: {key}")
            count = await asyncio.to_thread(self.compiler.token_count, messages)
            if count < 1:
                raise ValueError("Tokenizer returned an empty chat template")
            if self.client.cache_mode != CacheMode.DISABLED:
                await self.client.cache.put(key, count)
            return count

"""Direct DiffusionGemma decisions through vLLM, without a System One server."""

from __future__ import annotations

import asyncio
import json
import random
import string
from functools import partial

from odr.models import base, protocols, vllm_canvas, vllm_readout, vllm_types
from odr.models import client as http_clients
from odr.models.decisions import BINARY_INSTRUCTION, CHOICE_INSTRUCTION, message_size
from odr.models.types import Candidates, DecisionState, Scores, WireResponse
from odr.schema import VLLMDecisionArguments, fingerprint

ADAPTER_REVISION = vllm_canvas.ADAPTER_REVISION
TOP_LOGPROBS = 20
HEX_BASE = 16


class DecisionModelVLLM(base.DecisionModel):
    """Single-step exact-logprob binary and choice reads on a raw vLLM server."""

    def __init__(
        self,
        client: http_clients.CachedAsyncOpenAI,
        model_name: str,
        settings: VLLMDecisionArguments,
        tokenizer: protocols.DecisionTokenizer | None = None,
    ) -> None:
        """Configure explicit deployment identity and lazy tokenizer compilation.

        :param client: Shared client pointing to the raw vLLM ``/v1`` endpoint.
        :param model_name: Served alias of the DiffusionGemma weights.
        :param settings: Pinned tokenizer and deployment identity.
        :param tokenizer: Optional explicitly supplied compatible tokenizer.
        """
        self.client = client
        self.model_name = model_name
        self.settings = settings
        self.plans = vllm_canvas.TokenizerCache(client, settings, tokenizer)

    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Read independent yes/no decisions while preserving shared context.

        :param state: Query and conversation history.
        :param candidates: All candidate identities and descriptions in the batch.
        :returns: Native yes probabilities by candidate identity.
        """
        if not candidates:
            return {}
        plan = await self.plans.plan(("yes", "no"))
        probabilities = await asyncio.gather(
            *(
                self._read(
                    state,
                    candidates,
                    binary_question(text),
                    plan,
                )
                for text in candidates.values()
            )
        )
        return {
            identity: scores[0] for identity, scores in zip(candidates, probabilities)
        }

    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Read an exact distribution over a single shared-context shortlist.

        :param state: Query and conversation history.
        :param candidates: Ordered candidate identities and descriptions.
        :returns: Normalized option probabilities; empty input requires no work.
        :raises ValueError: If the shortlist exceeds the supported label alphabet.
        """
        if len(candidates) < 2:
            return {identity: 1.0 for identity in candidates}
        if len(candidates) > len(string.ascii_letters):
            raise ValueError("Direct vLLM readout supports at most 52 choices")
        labels = tuple(
            (string.ascii_uppercase + string.ascii_lowercase)[: len(candidates)]
        )
        plan = await self.plans.plan(labels)
        probabilities = await self._read(
            state, candidates, choice_question(labels, candidates), plan
        )
        return dict(zip(candidates, probabilities))

    def context_size(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int:
        """Measure native question and state messages without loading a tokenizer.

        :param state: Original query and conversation history.
        :param candidates: All candidates sharing the decision context.
        :param choice: Whether to measure a shortlist choice.
        :returns: Maximum rendered content size of the physical reads in characters.
        """
        state = {**state, "candidates": candidates}
        if choice:
            labels = tuple(
                (string.ascii_uppercase + string.ascii_lowercase)[: len(candidates)]
            )
            return message_size(
                read_messages(state, choice_question(labels, candidates))
            )
        return max(
            (
                message_size(read_messages(state, binary_question(text)))
                for text in candidates.values()
            ),
            default=0,
        )

    async def context_tokens(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int:
        """Count the complete pinned chat template and output without model inference.

        Counts are cached so replay never needs tokenizer downloads.

        :param state: Query and conversation history.
        :param candidates: Complete shared candidate context.
        :param choice: Whether to measure one shortlist choice.
        :returns: Maximum prompt-plus-output token count of a physical read.
        """
        if not candidates or (choice and len(candidates) == 1):
            return 0
        state = {**state, "candidates": candidates}
        labels, questions = read_questions(candidates, choice)
        plan = await self.plans.plan(labels)
        counts = await asyncio.gather(
            *(
                self.plans.count(read_messages(state, question))
                for question in questions
            )
        )
        return max(counts) + plan.output_length

    async def _read(
        self,
        state: DecisionState,
        candidates: Candidates,
        question: str,
        plan: vllm_types.CanvasPlan,
    ) -> list[float]:
        """Execute one physical read with full validation before caching.

        :param state: Original query and conversation history.
        :param candidates: Complete shared-context candidate mapping.
        :param question: Binary or choice instructions.
        :param plan: Tokenizer-derived canvas and exact-label mapping.
        :returns: Ordered label probabilities.
        """
        payload = self._payload({**state, "candidates": candidates}, question, plan)
        readout = partial(vllm_readout.read_probabilities, plan=plan)
        response = await self.client.request(
            "chat/completions",
            payload,
            context=[ADAPTER_REVISION, self.settings.model_dump(), tuple(candidates)],
            validator=readout,
        )
        return readout(response)

    def _payload(
        self, state: DecisionState, question: str, plan: vllm_types.CanvasPlan
    ) -> WireResponse:
        """Build the raw vLLM extension fields required by diffusion readout.

        :param state: Complete shared query context.
        :param question: Question and allowed options.
        :param plan: Tokenizer-derived fixed canvas and answer slot.
        :returns: Native chat completion payload with exact token-ID logprobs.
        """
        canvas = list(plan.canvas)
        identity = fingerprint([self.settings.seed, state, question])
        seed = int(identity[:8], HEX_BASE)
        canvas[plan.position] = random.Random(seed).randrange(plan.vocabulary_size)
        return {
            "model": self.model_name,
            "messages": read_messages(state, question),
            "max_tokens": plan.output_length,
            "logprobs": True,
            "top_logprobs": TOP_LOGPROBS,
            "logprob_token_ids": plan.label_ids,
            "return_tokens_as_token_ids": True,
            "chat_template_kwargs": {"enable_thinking": False},
            "vllm_xargs": {
                "diffusion_seed_canvas": canvas,
                "diffusion_canvas_length": len(canvas),
                "diffusion_max_steps": 1,
                "diffusion_read_only": True,
            },
        }


def choice_question(labels: tuple[str, ...], candidates: Candidates) -> str:
    """Render the ordered label-to-candidate mapping used by a choice read.

    :param labels: Tokenizer-validated answer labels.
    :param candidates: Ordered descriptions of candidate passages or branches.
    :returns: Choice instructions containing all alternatives.
    """
    options = "\n".join(
        f"{label}: {text}" for label, text in zip(labels, candidates.values())
    )
    return f"{CHOICE_INSTRUCTION}\n{options}"


def binary_question(text: str) -> str:
    """Render a binary relevance question with explicit yes/no labels.

    :param text: Candidate description.
    :returns: Native binary question instructions.
    """
    return f"{BINARY_INSTRUCTION}\nCandidate:\n{text}\nAllowed labels: yes, no."


def read_messages(state: DecisionState, question: str) -> list[dict[str, str]]:
    """Render the exact message content shared by dispatch and budgeting.

    :param state: Complete shared decision context.
    :param question: Native question and labels.
    :returns: Model instruction and serialized user context.
    """
    return [
        dict(
            role="system",
            content=f"{question}\nReply with q1: followed by one allowed label.",
        ),
        dict(role="user", content=json.dumps(state, ensure_ascii=False)),
    ]


def read_questions(
    candidates: Candidates, choice: bool
) -> tuple[tuple[str, ...], list[str]]:
    """Resolve the labels and question inputs for physical read budgeting.

    :param candidates: Complete candidate descriptions.
    :param choice: Whether to construct a shortlist choice.
    :returns: Ordered allowed labels and physical read questions.
    """
    if choice:
        labels = tuple(
            (string.ascii_uppercase + string.ascii_lowercase)[: len(candidates)]
        )
        return labels, [choice_question(labels, candidates)]
    return ("yes", "no"), [binary_question(text) for text in candidates.values()]

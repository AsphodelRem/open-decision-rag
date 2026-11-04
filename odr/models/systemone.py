"""Jev-compatible decision server adapter for razorback16/openjev."""

from __future__ import annotations

import json
from functools import partial
from typing import Any

from odr.models.base import DecisionModel
from odr.models.client import CachedAsyncOpenAI
from odr.models.decisions import (
    BINARY_INSTRUCTION,
    CHOICE_INSTRUCTION,
    CHOICE_QUESTION,
    systemone_scores,
)
from odr.models.types import Candidates, DecisionState, Scores, WireResponse

ADAPTER_REVISION = "systemone-v1"
MAX_CHOICES = 255


class DecisionModelSystemOne(DecisionModel):
    """Native binary and choice probabilities through a shared cached client."""

    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        *,
        deployment_revision: str = "unversioned",
        **settings: Any,
    ) -> None:
        """Bind an explicit server and model without loading local weights.

        :param client: Shared client with a base URL ending in ``/v1``.
        :param model_name: Model alias served by the System One endpoint.
        :param deployment_revision: Cache identity for server and weights versions.
        :param settings: Native inference options, such as ``samples`` or ``steps``.
        """
        self.client = client
        self.model_name = model_name
        self.deployment_revision = deployment_revision
        self.settings = settings

    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Ask independent relevance questions in a shared context.

        :param state: Query, conversation history and shared candidates.
        :param candidates: Candidate IDs mapped to rendered descriptions.
        :returns: Native yes probabilities by candidate ID.
        """
        return await self._judge(state, candidates, binary_questions(candidates))

    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Compare the complete shortlist in one choice question.

        :param state: Query, conversation history and shared candidates.
        :param candidates: Candidate IDs mapped to rendered descriptions.
        :returns: Native normalized candidate probabilities.
        :raises ValueError: If the shortlist exceeds the native option limit.
        """
        if len(candidates) > MAX_CHOICES:
            raise ValueError("System One supports at most 255 choice options")
        return await self._judge(
            state, candidates, choice_questions(candidates), choice=True
        )

    def context_size(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int:
        """Include native questions in the server-rendered context estimate.

        The server's private chat template remains outside this estimate.

        :param state: Query and conversation history.
        :param candidates: Complete candidate context.
        :param choice: Whether to include a choice option mapping.
        :returns: Serialized state and question characters.
        """
        questions = (
            choice_questions(candidates) if choice else binary_questions(candidates)
        )
        return len(
            json.dumps(
                dict(state={**state, "candidates": candidates}, questions=questions),
                ensure_ascii=False,
            )
        )

    async def _judge(
        self,
        state: DecisionState,
        candidates: Candidates,
        questions: WireResponse,
        choice: bool = False,
    ) -> Scores:
        """Validate responses before publishing them to the shared cache.

        :param state: Shared query context.
        :param candidates: Expected candidate identities.
        :param questions: Native question definitions.
        :param choice: Whether to read a normalized choice distribution.
        :returns: Validated candidate probabilities, or an empty mapping.
        """
        if not candidates:
            return {}
        payload = {
            **self.settings,
            "model": self.model_name,
            "state": {**state, "candidates": candidates},
            "questions": questions,
        }
        readout = partial(systemone_scores, candidates=candidates, choice=choice)
        response = await self.client.request(
            "systemone",
            payload,
            context=[ADAPTER_REVISION, self.deployment_revision, tuple(candidates)],
            validator=readout,
        )
        return readout(response)


def binary_questions(candidates: Candidates) -> WireResponse:
    """Render independent native questions exactly as sent to System One.

    :param candidates: Candidate identities and descriptions.
    :returns: Native question definitions.
    """
    return {
        identity: dict(
            type="noul", instructions=f"{BINARY_INSTRUCTION}\nCandidate:\n{text}"
        )
        for identity, text in candidates.items()
    }


def choice_questions(candidates: Candidates) -> WireResponse:
    """Render the native shortlist option map.

    :param candidates: Candidate identities and descriptions.
    :returns: Single native choice question.
    """
    return {
        CHOICE_QUESTION: dict(
            type="choice", instructions=CHOICE_INSTRUCTION, criteria=candidates
        )
    }

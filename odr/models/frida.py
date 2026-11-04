"""Native FRIDA-Decisions readout through its vLLM pooling plugin."""

import json
from functools import partial
from typing import Any

from pydantic import TypeAdapter

from odr.models.base import DecisionModel
from odr.models.client import CachedAsyncOpenAI
from odr.models.decisions import systemone_scores
from odr.models.systemone import binary_questions, choice_questions
from odr.models.types import Candidates, DecisionState, Scores, WireResponse

ADAPTER_REVISION = "frida-pooling-v1"
POOLING_DATA = TypeAdapter(dict[str, Any])


def frida_candidates(candidates: Candidates) -> Candidates:
    """Give model-visible options short neutral identities in source order.

    :param candidates: Original source identities and rendered descriptions.
    :returns: Consecutively numbered option identities and unchanged descriptions.
    """
    return {f"c{index}": text for index, text in enumerate(candidates.values())}


def frida_request(
    state: DecisionState, candidates: Candidates, choice: bool
) -> WireResponse:
    """Render shared context and native questions without backend-only options.

    :param state: Original query and conversation history.
    :param candidates: Source-ordered candidate descriptions.
    :param choice: Whether to compare the shortlist instead of independent yes/no.
    :returns: FRIDA request carried inside the pooling data envelope.
    """
    options = frida_candidates(candidates)
    references = {
        identity: f"Candidate {identity} in the shared state" for identity in options
    }
    return dict(
        state=dict(
            query=state["query"],
            history=state["history"],
            candidates=[
                dict(id=identity, text=text) for identity, text in options.items()
            ],
        ),
        questions=choice_questions(references)
        if choice
        else binary_questions(references),
    )


def frida_scores(
    response: WireResponse, candidates: Candidates, choice: bool = False
) -> Scores:
    """Validate the pooling envelope and restore caller identities before caching.

    :param response: Full vLLM pooling wire response.
    :param candidates: Expected original source identities in input order.
    :param choice: Whether to validate a normalized choice distribution.
    :returns: Native probabilities keyed by original identities.
    :raises ValueError: If data, scores or reported state truncation are invalid.
    """
    provider = POOLING_DATA.validate_python(response.get("data"), strict=True)
    usage = POOLING_DATA.validate_python(provider.get("usage", {}), strict=True)
    if usage.get("state_truncated", False) is not False:
        raise ValueError("FRIDA truncated the shared state; reduce the context budget")
    scores = systemone_scores(provider, frida_candidates(candidates), choice)
    return {identity: scores[f"c{index}"] for index, identity in enumerate(candidates)}


class DecisionModelFRIDA(DecisionModel):
    """Native independent relevance and choice probabilities from a FRIDA server."""

    def __init__(
        self,
        client: CachedAsyncOpenAI,
        model_name: str,
        *,
        deployment_revision: str,
    ) -> None:
        """Bind an explicit pooling deployment without importing local model code.

        :param client: Shared client using the server root URL, without ``/v1``.
        :param model_name: Model alias served by the FRIDA vLLM plugin.
        :param deployment_revision: Weights, plugin and server configuration identity.
        """
        self.client = client
        self.model_name = model_name
        self.deployment_revision = deployment_revision

    async def binary_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Send all independent relevance questions in one logical pooling request.

        :param state: Query and conversation history.
        :param candidates: Complete shared candidate batch in source order.
        :returns: Native yes probabilities without cross-candidate normalization.
        """
        return await self._judge(state, candidates)

    async def choice_judge(
        self, state: DecisionState, candidates: Candidates
    ) -> Scores:
        """Compare the shortlist through native choice probabilities.

        :param state: Query and conversation history.
        :param candidates: Complete shared shortlist in source order.
        :returns: Normalized probabilities; zero or one option requires no HTTP.
        """
        if len(candidates) < 2:
            return {identity: 1.0 for identity in candidates}
        return await self._judge(state, candidates, choice=True)

    def context_size(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> int:
        """Estimate rendered request characters, excluding server tokenizer suffixes.

        :param state: Query and conversation history.
        :param candidates: Shared candidate descriptions.
        :param choice: Whether to measure a choice request.
        :returns: Serialized state and questions character count, not a token count.
        """
        return len(
            json.dumps(frida_request(state, candidates, choice), ensure_ascii=False)
        )

    async def _judge(
        self, state: DecisionState, candidates: Candidates, choice: bool = False
    ) -> Scores:
        """Read and cache only complete, valid, untruncated responses.

        :param state: Query and conversation history.
        :param candidates: Expected source identities and descriptions.
        :param choice: Whether to request a choice distribution.
        :returns: Validated native probabilities in caller identity space.
        """
        if not candidates:
            return {}
        readout = partial(frida_scores, candidates=candidates, choice=choice)
        response = await self.client.request(
            "pooling",
            dict(model=self.model_name, data=frida_request(state, candidates, choice)),
            context=[ADAPTER_REVISION, self.deployment_revision, tuple(candidates)],
            validator=readout,
        )
        return readout(response)

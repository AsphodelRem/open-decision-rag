"""Validated templates and response schemas for direct vLLM reads."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, StrictInt, model_validator

from odr.schema.records import FrozenModel

LogProbability = Annotated[float, Field(strict=True, allow_inf_nan=False, le=0)]


class CanvasPlan(FrozenModel):
    """Tokenizer-derived answer slot persisted separately from wire responses.

    :ivar canvas: Fixed template, turn terminator and padding token IDs.
    :ivar position: Answer position in generated logprob rows.
    :ivar output_length: Number of output tokens through the turn terminator.
    :ivar label_ids: Ordered exact token IDs of all alternatives.
    :ivar labels: Original ordered answer labels used to validate cache plans.
    :ivar vocabulary_size: Exclusive upper bound for seeded answer noise.
    """

    canvas: list[StrictInt]
    position: int = Field(strict=True, ge=0)
    output_length: int = Field(strict=True, gt=0)
    label_ids: list[StrictInt] = Field(min_length=2)
    labels: list[str] = Field(min_length=2)
    vocabulary_size: int = Field(strict=True, gt=0)

    @model_validator(mode="after")
    def validate_tokens(self) -> CanvasPlan:
        """Reject invalid, duplicate and out-of-range template tokens.

        :returns: Validated plan.
        :raises ValueError: If a slot or token identity is invalid.
        """
        if not self.position < self.output_length <= len(self.canvas):
            raise ValueError("Answer position exceeds the canvas")
        distinct = len(set(self.label_ids)) == len(self.label_ids)
        if not distinct or len(self.labels) != len(self.label_ids):
            raise ValueError("Decision labels must have distinct token IDs")
        tokens = self.canvas + self.label_ids
        if any(token < 0 or token >= self.vocabulary_size for token in tokens):
            raise ValueError("Canvas token ID exceeds the tokenizer vocabulary")
        return self


class TokenLogprob(BaseModel):
    """Exact token identity and its finite natural log probability.

    :ivar token: vLLM token string in ``token_id:123`` notation.
    :ivar logprob: Finite nonpositive log probability.
    """

    token: str
    logprob: LogProbability


class LogprobRow(BaseModel):
    """Candidate evidence for a single generated position.

    :ivar top_logprobs: Returned token log probabilities, including requested IDs.
    """

    top_logprobs: list[TokenLogprob]


class ChatLogprobs(BaseModel):
    """Ordered raw output-token evidence.

    :ivar rows: Token rows read from the provider content field.
    """

    rows: list[LogprobRow] = Field(alias="content")


class VLLMChoice(BaseModel):
    """Required readout evidence in one chat completion choice.

    :ivar logprobs: Raw output-token log probabilities.
    """

    logprobs: ChatLogprobs


class VLLMResponse(BaseModel):
    """Response envelope that retains only fields needed for readout.

    :ivar choices: Provider choices; the adapter requires exactly one.
    """

    choices: list[VLLMChoice] = Field(min_length=1, max_length=1)


class PromptCount(FrozenModel):
    """Validated cached chat-template token count.

    :ivar count: Positive exact prompt token count.
    """

    count: int = Field(strict=True, gt=0)

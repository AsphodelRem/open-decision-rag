"""Shared model input types independent of concrete providers."""

from enum import Enum
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

Messages = list[dict[str, str]]
Vectors = list[list[float]]
Candidates = dict[str, str]
Scores = dict[str, float]
WireResponse = dict[str, Any]


class DecisionState(TypedDict):
    """Shared query context supplied to a decision model.

    :ivar query: Original validated query.
    :ivar history: Ordered conversation messages.
    :ivar candidates: Candidate IDs and rendered descriptions.
    """

    query: str
    history: tuple[tuple[str, str], ...]
    candidates: Candidates


Probability = Annotated[
    float,
    Field(strict=True, ge=0, le=1, allow_inf_nan=False),
]


class BinaryAnswer(BaseModel):
    """Native yes probability; confidence is not part of the readout.

    :ivar type: Native binary question discriminator.
    :ivar noul: Probability of a positive answer.
    """

    type: Literal["noul"]
    noul: Probability


class ChoiceAnswer(BaseModel):
    """Native choice distribution with original candidate identities.

    :ivar type: Native choice question discriminator.
    :ivar choice: Selected candidate identity.
    :ivar probabilities: Probability for each candidate.
    """

    type: Literal["choice"]
    choice: str
    probabilities: dict[str, Probability]


class ChatMessage(BaseModel):
    """Required text field of a decision chat completion.

    :ivar text: Generated JSON object, read from the provider content field.
    """

    text: str = Field(alias="content")


class ChatChoice(BaseModel):
    """Required message field of a chat choice.

    :ivar message: Generated assistant message.
    """

    message: ChatMessage


class ChatCompletion(BaseModel):
    """Required choices field of a decision chat completion.

    :ivar choices: Generated alternatives in provider order.
    """

    choices: list[ChatChoice]


class CacheMode(str, Enum):
    """Explicit supported cache execution policies."""

    READ_WRITE = "read_write"
    REFRESH = "refresh"
    DISABLED = "disabled"
    REPLAY = "replay"

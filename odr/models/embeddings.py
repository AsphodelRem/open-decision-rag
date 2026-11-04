"""Strict embedding response validation before cache publication."""

from __future__ import annotations

from typing import Annotated

from pydantic import BaseModel, Field, TypeAdapter

from odr.models.types import Vectors, WireResponse

FiniteComponent = Annotated[float, Field(strict=True, allow_inf_nan=False)]
VECTOR_ADAPTER = TypeAdapter(list[list[FiniteComponent]])


class EmbeddingRecord(BaseModel):
    """One indexed finite provider vector.

    :ivar index: Nonnegative original batch position.
    :ivar embedding: Finite numeric vector.
    """

    index: int = Field(strict=True, ge=0)
    embedding: list[FiniteComponent] = Field(min_length=1)


class EmbeddingResponse(BaseModel):
    """Required provider embedding records.

    :ivar records: Indexed vectors read from the provider data field.
    """

    records: list[EmbeddingRecord] = Field(alias="data")


def validate_vectors(vectors: Vectors, count: int) -> Vectors:
    """Reject invalid counts, dimensions, numeric types and nonfinite values.

    :param vectors: Ordered embedding vectors.
    :param count: Exact expected input count.
    :returns: Strictly validated vectors.
    :raises ValueError: If a vector or its dimensions are invalid.
    """
    vectors = VECTOR_ADAPTER.validate_python(vectors, strict=True)
    if not vectors or len(vectors) != count:
        raise ValueError("Invalid embedding count or empty vector")
    dimension = len(vectors[0])
    if not dimension or any(len(vector) != dimension for vector in vectors):
        raise ValueError("Invalid embedding dimensions")
    return vectors


def decode_embeddings(response: WireResponse, count: int) -> Vectors:
    """Require complete, unique batch indices before accepting a wire response.

    :param response: Complete provider response.
    :param count: Number of original batch inputs.
    :returns: Validated vectors reordered to match inputs.
    :raises ValueError: If indices, vectors or response fields are invalid.
    """
    records = sorted(
        EmbeddingResponse.model_validate(response, strict=True).records,
        key=lambda record: record.index,
    )
    if [record.index for record in records] != list(range(count)):
        raise ValueError("Missing or duplicate embedding indices")
    return validate_vectors([record.embedding for record in records], count)

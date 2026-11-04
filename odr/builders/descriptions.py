"""Validated descriptions for hierarchy groups."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

    from odr.models.base import LLM
DEFAULT_ITERATIONS = 300
DESCRIPTION_WORD_LIMIT = 30


class LLMDescriptionBuilder:
    """Bounded cached generation of validated short group descriptions."""

    def __init__(
        self,
        llm: LLM,
        max_words: int = DESCRIPTION_WORD_LIMIT,
        attempts: int = 2,
    ) -> None:
        """Configure description validation and bounded repair.

        :param llm: Injected generation model.
        :param max_words: Maximum words in an accepted description.
        :param attempts: Maximum description generation and repair attempts.
        """
        self.llm = llm
        self.max_words = max_words
        self.attempts = attempts

    async def describe(self, titles: Sequence[str]) -> str:
        """Fail the build rather than publishing an invalid summary.

        :param titles: Document titles ordered by cluster typicality.
        :returns: A validated short group description.
        :raises ValueError: If the input, stored data, or operation result is invalid.
        """
        instruction = f"Describe dominant and minority themes in one sentence of at most {self.max_words} words."
        messages = [
            dict(role="system", content=instruction),
            dict(role="user", content="\n".join(titles[:DEFAULT_ITERATIONS])),
        ]
        attempt = 0
        while attempt < self.attempts:
            description = (await self.llm.chat_completion(messages)).strip()
            if (
                description
                and "\n" not in description
                and (len(description.split()) <= self.max_words)
            ):
                return description
            attempt += 1
            messages.extend(
                [
                    dict(role="assistant", content=description),
                    dict(
                        role="user",
                        content=f"Repair invalid description. {instruction}",
                    ),
                ]
            )
        raise ValueError("Description generation failed validation")

"""Source coordinate and tokenizer callable contracts."""

from collections.abc import Callable, Iterable

TokenSpan = tuple[int, int]
SourceOffsets = tuple[int | None, int | None]
Tokenizer = Callable[[str], Iterable[TokenSpan]]

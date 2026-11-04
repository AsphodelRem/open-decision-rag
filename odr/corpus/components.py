"""Typed and runtime-checked corpus constructor components."""

from typing import TypedDict

from odr.models.base import LLM, Embedder
from odr.schema.arguments import BuilderArguments, StorageArguments


class CorpusComponents(TypedDict, total=False):
    """Supported injected corpus configuration.

    :ivar embedder: Document embedding model.
    :ivar llm: Cluster-description generator.
    :ivar builder_settings: Validated semantic hierarchy settings.
    :ivar storage_settings: Validated publication settings.
    """

    embedder: Embedder
    llm: LLM
    builder_settings: BuilderArguments
    storage_settings: StorageArguments


def validate_components(components: CorpusComponents) -> None:
    """Reject misspelled constructor keywords instead of silently ignoring them.

    :param components: Injected corpus component mapping.
    :raises TypeError: If a keyword is unsupported.
    """
    unknown = set(components) - {
        "embedder",
        "llm",
        "builder_settings",
        "storage_settings",
    }
    if unknown:
        raise TypeError(f"Unknown corpus components: {sorted(unknown)}")

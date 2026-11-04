"""Local embeddings with lazy model loading and persistent result caching."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from odr.models.protocols import LocalEncoder
    from odr.models.types import Vectors
    from odr.storage.cache import CacheBackend
import asyncio

from odr.models.base import Embedder
from odr.models.openai import validate_vectors
from odr.schema import fingerprint
from odr.storage.cache import MemoryCache

EMBEDDING_BATCH_SIZE = 32


class EmbedderSentenceTransformers(Embedder):
    """CPU/GPU encoding off the event loop."""

    def __init__(
        self, model_name: str, cache: CacheBackend | None = None, **settings: Any
    ) -> None:
        """Configure lazy local encoding and its persistent cache.

        :param model_name: Model identity or served model alias.
        :param cache: Shared response cache; defaults to an in-memory cache.
        :param settings: Provider options forwarded to the selected backend.
        """
        self.model_name = model_name
        self.cache = MemoryCache() if cache is None else cache
        self.settings = settings
        self.model: LocalEncoder | None = None
        self.lock = asyncio.Lock()

    async def embed(self, texts: Sequence[str]) -> Vectors:
        """Cache exact float JSON vectors including model configuration.

        :param texts: Ordered source texts to embed.
        :returns: Embedding vectors in input order.
        """
        key = fingerprint(["local-embedding-v1", self.model_name, self.settings, texts])
        async with self.lock:
            vectors: Vectors | None = await self.cache.get(key)
            if vectors is None:
                vectors = await asyncio.to_thread(self._encode, texts)
                validate_vectors(vectors, len(texts))
                await self.cache.put(key, vectors)
        return validate_vectors(vectors, len(texts))

    def _encode(self, texts: Sequence[str]) -> Vectors:
        """Load the local embedding model lazily and encode source texts.

        :param texts: Ordered source texts to embed.
        :returns: Embedding vectors converted to Python scalars.
        """
        if self.model is None:
            factory = cast(
                "Callable[..., LocalEncoder]",
                import_module("sentence_transformers").SentenceTransformer,
            )
            self.model = factory(
                self.model_name,
                revision=self.settings.get("revision"),
                device=self.settings.get("device", "cpu"),
            )
        prefix = self.settings.get("prefix", "query: ")
        return self.model.encode(
            [prefix + text for text in texts],
            normalize_embeddings=self.settings.get("normalize", True),
            batch_size=self.settings.get("batch_size", EMBEDDING_BATCH_SIZE),
        ).tolist()

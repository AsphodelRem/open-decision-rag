"""One execution layer for model HTTP calls and local operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from odr.models.types import WireResponse
    from odr.storage.cache import CacheBackend
import asyncio
from collections import Counter

import httpx

from odr.models import diagnostics, transport
from odr.schema import fingerprint
from odr.storage.cache import CacheMissError, MemoryCache

SERVER_ERROR_STATUS = 500
RATE_LIMIT_STATUS = 429


class CachedAsyncOpenAI(transport.HTTPExecution, transport.CachePolicy):
    """Explicit endpoint, persistent cache, retries and shared concurrency."""

    def __init__(
        self,
        base_url: str,
        cache: CacheBackend | None = None,
        cache_mode: str = "read_write",
        **settings: Any,
    ) -> None:
        """Configure shared transport, cache and concurrency ownership.

        :param base_url: Explicit OpenAI-compatible API base URL.
        :param cache: Shared response cache; defaults to an in-memory cache.
        :param cache_mode: Request-cache read and write policy.
        :param settings: Provider options forwarded to the selected backend.
        """
        self._base_url = base_url.rstrip("/")
        self.cache = MemoryCache() if cache is None else cache
        self.cache_mode = cache_mode
        self.settings = dict(settings)
        self._semaphore = asyncio.Semaphore(settings.get("rate_max_simultaneous", 8))
        self._pending: dict[tuple[str, str], asyncio.Task[WireResponse]] = {}
        self.stats: Counter[str] = Counter()
        self._transport = settings.get("http_client") or httpx.AsyncClient(
            timeout=settings.get("timeout", 60)
        )
        self._owns_transport = "http_client" not in settings

    async def __aenter__(self) -> CachedAsyncOpenAI:
        """Enter the component context.

        :returns: The opened object.
        """
        return self

    async def __aexit__(self, *exception: object) -> None:
        """Exit the component context according to resource ownership.

        :param exception: Exception context or provider failure being handled.
        """
        await asyncio.gather(*self._pending.values(), return_exceptions=True)
        if self._owns_transport:
            await self._transport.aclose()

    async def request(
        self,
        operation: str,
        payload: WireResponse,
        context: object = None,
        validator: Callable[[WireResponse], object] | None = None,
    ) -> WireResponse:
        """Cache full wire responses; identity excludes credentials.

        :param operation: Endpoint path relative to the configured API base URL.
        :param payload: Semantic operation input used for serialization or transport.
        :param context: Semantic cache context or shared hierarchy construction state.
        :param validator: Validation callback run before a response is accepted.
        :returns: The full validated provider response.
        :raises CacheMissError: If the input, stored data, or operation result is invalid.
        """
        identity = [
            1,
            self._base_url,
            self.settings.get("cache_scope"),
            operation,
            payload,
            context,
        ]
        key = fingerprint(identity)
        with diagnostics.RequestScope(operation, key):
            return await self._request(operation, payload, key, validator)

    async def _request(
        self,
        operation: str,
        payload: WireResponse,
        key: str,
        validator: Callable[[WireResponse], object] | None,
    ) -> WireResponse:
        """Execute cache lookup and request singleflight within one diagnostic scope.

        :param operation: Endpoint operation.
        :param payload: Semantic provider request.
        :param key: Complete response-cache identity.
        :param validator: Provider response validator.
        :returns: Validated cached or fresh wire response.
        """
        self.stats["logical_calls"] += 1
        if self.cache_mode not in {"disabled", "refresh"}:
            cached = await self.cache.get(key)
            if cached is not None:
                diagnostics.record_received(cached, cached=True)
                if validator is not None:
                    validator(cached)
                diagnostics.record_response(cached, cached=True)
                self.stats["hits"] += 1
                return cached
        if self.cache_mode == "replay":
            raise CacheMissError(key)
        self.stats["misses"] += 1
        flight_key = (key, self.cache_mode)
        task = self._pending.get(flight_key)
        if task is None:
            task = asyncio.create_task(
                self._execute(operation, payload, key, validator)
            )
            self._pending[flight_key] = task
            task.add_done_callback(
                lambda completed: self._pending.pop(flight_key, None)
            )
        else:
            diagnostics.record_shared()
            self.stats["singleflight_waits"] += 1
        response = await asyncio.shield(task)
        diagnostics.record_response(response)
        return response

    async def _execute(
        self,
        operation: str,
        payload: WireResponse,
        key: str,
        validator: Callable[[WireResponse], object] | None,
    ) -> WireResponse:
        """Validate and persist one successful physical model operation.

        :param operation: Endpoint path relative to the configured API base URL.
        :param payload: Semantic operation input used for serialization or transport.
        :param key: Semantic request cache key.
        :param validator: Validation callback run before a response is accepted.
        :returns: The successful provider response after persistence.
        """
        response = await self._retry(operation, payload)
        diagnostics.record_received(response)
        if validator is not None:
            validator(response)
        diagnostics.record_response(response)
        if self.cache_mode != "disabled":
            await self.cache.put(key, response, self.settings.get("ttl"))
        self.stats["actual_inference_calls"] += 1
        return response

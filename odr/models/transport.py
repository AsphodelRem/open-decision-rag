"""Internal HTTPExecution operations."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

    from odr.models.types import WireResponse
import asyncio

import httpx

from odr.models.diagnostics import CURRENT_CALL
from odr.models.types import CacheMode

SERVER_ERROR_STATUS = 500
RATE_LIMIT_STATUS = 429


class HTTPExecution:
    """Shared HTTP transport state and bounded retry execution.

    :ivar settings: Provider options shared by physical HTTP attempts.
    :ivar stats: Transport and request execution counters.
    """

    settings: dict[str, Any]
    stats: Counter[str]
    _semaphore: asyncio.Semaphore
    _transport: httpx.AsyncClient
    _base_url: str

    def semantic_identity(self) -> dict[str, object]:
        """Expose serving identity without credentials or mutable cache policy.

        :returns: Endpoint and explicit tenant/deployment cache scope.
        """
        endpoint = httpx.URL(self._base_url).copy_with(username=None, password=None)
        return dict(endpoint=str(endpoint), scope=self.settings.get("cache_scope"))

    async def _retry(self, operation: str, payload: WireResponse) -> WireResponse:
        """Execute HTTP attempts within the configured retry budget.

        :param operation: Endpoint path relative to the configured API base URL.
        :param payload: Semantic operation input used for serialization or transport.
        :returns: The successful provider response within the retry budget.
        """
        attempt = 0
        while True:
            try:
                return await self._checked_post(operation, payload)
            except httpx.HTTPStatusError as exception:
                delay = self._retry_delay(exception, attempt)
            except httpx.TransportError:
                if attempt + 1 >= self.settings.get("max_attempts", 3):
                    raise
                delay = 2**attempt
            self.stats["retries"] += 1
            attempt += 1
            await asyncio.sleep(delay)

    async def _checked_post(
        self, operation: str, payload: WireResponse
    ) -> WireResponse:
        """Validate HTTP status before decoding the provider response.

        :param operation: Endpoint path relative to the configured API base URL.
        :param payload: Semantic operation input used for serialization or transport.
        :returns: The decoded response after HTTP status validation.
        """
        response = await self._post(operation, payload)
        response.raise_for_status()
        return response.json()

    def _retry_delay(self, exception: httpx.HTTPStatusError, attempt: int) -> float:
        """Apply retry classification and derive the provider retry delay.

        :param exception: Exception context or provider failure being handled.
        :param attempt: Zero-based retry attempt.
        :returns: Retry delay in seconds.
        """
        status = exception.response.status_code
        if status != RATE_LIMIT_STATUS and status < SERVER_ERROR_STATUS:
            raise exception
        if attempt + 1 >= self.settings.get("max_attempts", 3):
            raise exception
        return float(exception.response.headers.get("Retry-After", 2**attempt))

    async def _post(self, operation: str, payload: WireResponse) -> httpx.Response:
        """Send one HTTP attempt under the shared concurrency limit.

        :param operation: Endpoint path relative to the configured API base URL.
        :param payload: Semantic operation input used for serialization or transport.
        :returns: The raw provider HTTP response.
        """
        headers = {}
        api_key = self.settings.get("api_key")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        async with self._semaphore:
            self.stats["network_attempts"] += 1
            call = CURRENT_CALL.get()
            if call is not None:
                call.attempts += 1
            return await self._transport.post(
                f"{self._base_url}/{operation}", json=payload, headers=headers
            )


class CachePolicy:
    """Runtime validation for constructor and subsequent cache-mode changes."""

    @property
    def cache_mode(self) -> str:
        """Return the validated cache execution mode.

        :returns: Current cache policy name.
        """
        return self._cache_mode.value

    @cache_mode.setter
    def cache_mode(self, mode: str) -> None:
        """Reject unknown policies before any HTTP operation can be created.

        :param mode: Requested cache policy.
        :raises ValueError: If the policy is unsupported.
        """
        self._cache_mode = CacheMode(mode)

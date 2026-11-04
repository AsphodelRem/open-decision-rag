"""Run-scoped HTTP diagnostics isolated across concurrent asynchronous searches."""

from __future__ import annotations

import time
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any


@dataclass
class ModelCall:
    """One logical request and its physical transport activity.

    :ivar operation: Model endpoint operation.
    :ivar cache_key: Semantic response-cache identity.
    :ivar cache_hit: Whether the response came from cache.
    :ivar shared: Whether this caller waited on another in-process request.
    :ivar attempts: Number of physical HTTP attempts owned by this request.
    :ivar accepted: Whether a wire response passed validation.
    :ivar elapsed_seconds: Logical request wall time including retries or waiting.
    :ivar usage: Provider-reported usage retained without payload text.
    :ivar error_type: Exception class on failure, without sensitive request details.
    """

    operation: str
    cache_key: str
    cache_hit: bool = False
    shared: bool = False
    attempts: int = 0
    accepted: bool = False
    elapsed_seconds: float = 0
    usage: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None


class ModelDiagnostics:
    """Collect model activity for an explicitly owned run, without global counters."""

    def __init__(self) -> None:
        """Initialize an empty run collector."""
        self.calls: list[ModelCall] = []
        self._token: Token[ModelDiagnostics | None] | None = None

    def __enter__(self) -> ModelDiagnostics:
        """Activate the collector for this asynchronous context.

        :returns: This run collector.
        """
        if self._token is not None:
            raise RuntimeError(
                "ModelDiagnostics is already active; use a separate collector per run"
            )
        self._token = CURRENT_RUN.set(self)
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Restore the surrounding run without suppressing errors.

        :param exception_type: Raised exception type.
        :param exception: Raised exception instance.
        :param traceback: Original traceback.
        """
        if self._token is not None:
            CURRENT_RUN.reset(self._token)
            self._token = None


CURRENT_RUN: ContextVar[ModelDiagnostics | None] = ContextVar(
    "decisionrag_run", default=None
)
CURRENT_CALL: ContextVar[ModelCall | None] = ContextVar(
    "decisionrag_call", default=None
)


class RequestScope:
    """Track a logical request while its physical task inherits the same context."""

    def __init__(self, operation: str, key: str) -> None:
        """Bind endpoint and cache identity without retaining sensitive payloads.

        :param operation: Requested model endpoint operation.
        :param key: Semantic cache key.
        """
        self.call = ModelCall(operation, key)
        self._token: Token[ModelCall | None] | None = None
        self.started: float = 0

    def __enter__(self) -> ModelCall:
        """Start request accounting inside the current optional run.

        :returns: Mutable diagnostics owned by this logical request.
        """
        self.started = time.perf_counter()
        self._token = CURRENT_CALL.set(self.call)
        run = CURRENT_RUN.get()
        if run is not None:
            run.calls.append(self.call)
        return self.call

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record elapsed time and restore the parent model call context.

        :param exception_type: Raised exception type.
        :param exception: Raised exception instance.
        :param traceback: Original traceback.
        """
        self.call.elapsed_seconds = time.perf_counter() - self.started
        self.call.error_type = exception_type.__name__ if exception_type else None
        if self._token is not None:
            CURRENT_CALL.reset(self._token)


def record_response(response: dict[str, Any], cached: bool = False) -> None:
    """Record validated usage without retaining provider prompt or response text.

    :param response: Validated provider wire response.
    :param cached: Whether the accepted response came from the request cache.
    """
    call = CURRENT_CALL.get()
    if call is not None:
        call.cache_hit = call.cache_hit or cached
        call.accepted = True
        call.usage = response.get("usage", {})


def record_received(response: dict[str, Any], cached: bool = False) -> None:
    """Retain physical usage even if semantic validation subsequently rejects it.

    :param response: Received wire response that has not yet passed validation.
    :param cached: Whether a previous cache entry supplied the wire response.
    """
    call = CURRENT_CALL.get()
    if call is not None:
        call.cache_hit = cached
        call.usage = response.get("usage", {})


def record_shared() -> None:
    """Mark a logical waiter whose physical request belongs to another caller."""
    call = CURRENT_CALL.get()
    if call is not None:
        call.shared = True

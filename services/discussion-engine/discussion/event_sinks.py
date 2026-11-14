"""Delivery adapters for discussion lifecycle events."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from threading import Lock
from typing import ClassVar, Protocol

import httpx

from .events import DiscussionEvent, utc_now


@dataclass(frozen=True)
class DeliveryResult:
    status: str
    attempts: int
    attempted_at: str
    last_error: str | None = None
    status_code: int | None = None

    @property
    def delivered(self) -> bool:
        return self.status == "delivered"


class DiscussionEventSink(Protocol):
    def publish(self, event: DiscussionEvent) -> DeliveryResult: ...

    def close(self) -> None: ...


class EventDeliveryError(RuntimeError):
    """Raised only when the caller explicitly requires event delivery."""


class NullEventSink:
    def publish(self, event: DiscussionEvent) -> DeliveryResult:
        return DeliveryResult(status="skipped", attempts=0, attempted_at=utc_now())

    def close(self) -> None:
        return None


class HttpEventSink:
    RETRYABLE_STATUS_CODES: ClassVar[set[int]] = {408, 429, 500, 502, 503, 504}

    def __init__(
        self,
        endpoint: str,
        *,
        token: str | None = None,
        timeout_seconds: float = 10.0,
        max_attempts: int = 5,
        retry_initial_seconds: float = 1.0,
        retry_max_seconds: float = 30.0,
        client: httpx.Client | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ):
        self.endpoint = endpoint
        self.token = token
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max(1, max_attempts)
        self.retry_initial_seconds = max(0.0, retry_initial_seconds)
        self.retry_max_seconds = max(0.0, retry_max_seconds)
        self.client = client
        self._owns_client = client is None
        self._client_lock = Lock()
        self.sleeper = sleeper

    def publish(self, event: DiscussionEvent) -> DeliveryResult:
        return self._publish_with_client(event, self._get_client())

    def _get_client(self) -> httpx.Client:
        if self.client is None:
            with self._client_lock:
                if self.client is None:
                    self.client = httpx.Client(timeout=self.timeout_seconds)
        return self.client

    def close(self) -> None:
        """Close only a client created by this sink, never an injected client."""
        if not self._owns_client:
            return
        with self._client_lock:
            client, self.client = self.client, None
        if client is not None:
            client.close()

    def _publish_with_client(
        self,
        event: DiscussionEvent,
        client: httpx.Client,
    ) -> DeliveryResult:
        headers = {
            "Content-Type": "application/json",
            "Idempotency-Key": event.event_id,
        }
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"

        last_error: str | None = None
        status_code: int | None = None
        attempted_at = utc_now()
        for attempt in range(1, self.max_attempts + 1):
            attempted_at = utc_now()
            try:
                response = client.post(
                    self.endpoint,
                    json=event.to_dict(),
                    headers=headers,
                    timeout=self.timeout_seconds,
                )
                status_code = response.status_code
                if 200 <= status_code < 300:
                    return DeliveryResult(
                        status="delivered",
                        attempts=attempt,
                        attempted_at=attempted_at,
                        status_code=status_code,
                    )
                last_error = f"consumer returned HTTP {status_code}"
                if status_code not in self.RETRYABLE_STATUS_CODES:
                    break
            except httpx.RequestError as exc:
                last_error = f"{type(exc).__name__}: {exc}"

            if attempt < self.max_attempts:
                delay = min(
                    self.retry_initial_seconds * (2 ** (attempt - 1)),
                    self.retry_max_seconds,
                )
                self.sleeper(delay)

        return DeliveryResult(
            status="pending",
            attempts=attempt,
            attempted_at=attempted_at,
            last_error=last_error,
            status_code=status_code,
        )


def build_event_sink(settings: object) -> DiscussionEventSink:
    endpoint = getattr(settings, "event_endpoint", None)
    if not endpoint:
        return NullEventSink()
    token = getattr(settings, "event_token", None)
    if hasattr(token, "get_secret_value"):
        token = token.get_secret_value()
    return HttpEventSink(
        endpoint,
        token=token,
        timeout_seconds=getattr(settings, "event_timeout_seconds", 10.0),
        max_attempts=getattr(settings, "event_max_attempts", 5),
        retry_initial_seconds=getattr(settings, "event_retry_initial_seconds", 1.0),
        retry_max_seconds=getattr(settings, "event_retry_max_seconds", 30.0),
    )

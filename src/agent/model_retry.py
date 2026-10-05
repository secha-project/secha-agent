"""Retry transient model requests without replaying successful agent/tool work."""

import asyncio
import logging
import random
import time
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from threading import Lock

import httpx
from openai import AsyncOpenAI
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.zai import ZaiProvider

logger = logging.getLogger(__name__)
TRANSIENT_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504}


class RequestPacer:
    """Space request starts and share cooldowns across worker event loops."""

    def __init__(self, min_interval_seconds: float = 0) -> None:
        self.min_interval_seconds = min_interval_seconds
        self._lock = Lock()
        self._next_start = 0.0

    def defer(self, seconds: float) -> None:
        with self._lock:
            self._next_start = max(self._next_start, time.monotonic() + seconds)

    async def wait(self) -> None:
        while True:
            with self._lock:
                now = time.monotonic()
                delay = self._next_start - now
                if delay <= 0:
                    self._next_start = now + self.min_interval_seconds
                    return
            # Recheck after sleeping: another worker may extend the cooldown.
            await asyncio.sleep(delay)


def retry_after_seconds(response: httpx.Response) -> float | None:
    """Read server delays without shortening a valid Retry-After value."""
    value = response.headers.get("retry-after")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        try:
            date = parsedate_to_datetime(value)
            if date.tzinfo is None:
                date = date.replace(tzinfo=UTC)
            seconds = (date - datetime.now(UTC)).total_seconds()
        except (ValueError, TypeError, OverflowError):
            return None
    return seconds if 0 <= seconds < float("inf") else None


class ModelRetryTransport(httpx.AsyncBaseTransport):
    def __init__(
        self,
        wrapped: httpx.AsyncBaseTransport | None = None,
        *,
        max_attempts: int | None = 6,
        retry_budget_seconds: float | None = 300,
        pacer: RequestPacer | None = None,
    ) -> None:
        self.wrapped = wrapped if wrapped is not None else httpx.AsyncHTTPTransport()
        self.max_attempts = max_attempts
        self.retry_budget_seconds = retry_budget_seconds
        self.pacer = pacer if pacer is not None else RequestPacer()

    def _exhausted(self, attempt: int, started: float, delay: float) -> bool:
        return (self.max_attempts is not None and attempt >= self.max_attempts) or (
            self.retry_budget_seconds is not None
            and time.monotonic() - started + delay > self.retry_budget_seconds
        )

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        # Bound pacing waits and in-flight non-streaming reads as well as backoff.
        # A zero retry budget retains its existing meaning: one initial attempt.
        if not self.retry_budget_seconds:
            return await self._request_with_retries(request)
        try:
            async with asyncio.timeout(self.retry_budget_seconds):
                return await self._request_with_retries(request)
        except TimeoutError as exc:
            raise httpx.ReadTimeout(
                "Model request retry budget exhausted", request=request
            ) from exc

    async def _request_with_retries(self, request: httpx.Request) -> httpx.Response:
        started = time.monotonic()
        attempt = 0
        while True:
            await self.pacer.wait()
            attempt += 1
            delay = min(15 * 2 ** min(attempt - 1, 5), 300) + random.uniform(0, 5)
            response: httpx.Response | None = None
            try:
                response = await self.wrapped.handle_async_request(request)
                content_type = response.headers.get("content-type", "").partition(";")[
                    0
                ]
                if (
                    response.status_code in TRANSIENT_STATUS_CODES
                    or content_type.strip().lower() != "text/event-stream"
                ):
                    # Include non-streaming body read failures in the retry policy.
                    # Never replay an SSE stream after exposing partial output.
                    await response.aread()
            except BaseException as exc:
                if response is not None:
                    await response.aclose()
                    response = None
                if not isinstance(exc, httpx.TransportError):
                    raise
                self.pacer.defer(delay)
                if self._exhausted(attempt, started, delay):
                    raise
            else:
                if response.status_code not in TRANSIENT_STATUS_CODES:
                    return response
                # Buffered errors retain their status/body for the SDK when
                # retries are exhausted.
                if response.status_code == 429:
                    try:
                        error = response.json().get("error", {})
                        code = error.get("code") if isinstance(error, dict) else None
                    except (ValueError, AttributeError):
                        code = None
                    if code in {"insufficient_quota", "billing_hard_limit_reached"}:
                        return response
                server_delay = retry_after_seconds(response)
                if server_delay is not None:
                    delay = max(delay, server_delay)
                self.pacer.defer(delay)
                if self._exhausted(attempt, started, delay):
                    return response
                await response.aclose()
            logger.warning(
                "Transient model request failure (%s); retry %d/%s in at least %.1fs",
                response.status_code if response is not None else "connection",
                attempt + 1,
                self.max_attempts if self.max_attempts is not None else "unlimited",
                delay,
            )

    async def aclose(self) -> None:
        await self.wrapped.aclose()


class _RetryingProviderMixin:
    """Keep Pydantic AI's client lifecycle and use one HTTP retry policy."""

    _default_base_url: str | None = None

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        max_attempts: int | None = 6,
        retry_budget_seconds: float | None = 300,
        min_interval_seconds: float = 0,
        pacer: RequestPacer | None = None,
    ) -> None:
        request_pacer = (
            pacer if pacer is not None else RequestPacer(min_interval_seconds)
        )

        def create_http_client() -> httpx.AsyncClient:
            return httpx.AsyncClient(
                transport=ModelRetryTransport(
                    max_attempts=max_attempts,
                    retry_budget_seconds=retry_budget_seconds,
                    pacer=request_pacer,
                ),
                timeout=httpx.Timeout(600, connect=5),
            )

        http_client = create_http_client()
        super().__init__(
            openai_client=AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or self._default_base_url,
                http_client=http_client,
                max_retries=0,
            )
        )
        self._own_http_client = http_client
        self._http_client_factory = create_http_client


class RetryingOpenAIProvider(_RetryingProviderMixin, OpenAIProvider):
    """OpenAI provider with shared pacing and transient request retries."""


class RetryingZaiProvider(_RetryingProviderMixin, ZaiProvider):
    """Native Z.ai profiles with the same client lifecycle and retry policy."""

    _default_base_url = "https://api.z.ai/api/paas/v4"

    @property
    def base_url(self) -> str:
        # ZaiProvider's constant URL would misidentify a configured endpoint.
        return str(self.client.base_url)

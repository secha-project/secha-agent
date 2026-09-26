import asyncio
import json
from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from itertools import pairwise
from types import SimpleNamespace

import httpx
import pytest
from openai import AsyncOpenAI, RateLimitError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.zai import ZaiProvider

import src.agent.model_retry as retries
from src.agent.config import AgentConfig, OpenAIProviderConfig
from src.agent.model_retry import (
    ModelRetryTransport,
    RequestPacer,
    RetryingOpenAIProvider,
    RetryingZaiProvider,
    retry_after_seconds,
)


@pytest.fixture
def delays(monkeypatch):
    recorded = []
    clock = [0.0]

    async def sleep(seconds):
        recorded.append(seconds)
        clock[0] += seconds

    monkeypatch.setattr(retries.asyncio, "sleep", sleep)
    monkeypatch.setattr(retries.random, "uniform", lambda *_: 0)
    # Patch this module's clock, without changing the asyncio event loop's clock.
    monkeypatch.setattr(retries, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    return recorded


@pytest.mark.parametrize("status", [408, 409, 429, 500, 502, 503, 504])
def test_transient_errors_retry_the_same_request(status, delays):
    bodies = []

    def respond(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(status, headers={"Retry-After": "45"}, json={})
        if len(bodies) == 2:
            return httpx.Response(status, json={})
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(httpx.MockTransport(respond))
        ) as client:
            result = await client.post("https://model.invalid", json={"prompt": "test"})
            assert result.json() == {"ok": True}

    asyncio.run(run())
    assert len(bodies) == 3 and len(set(bodies)) == 1
    assert delays == [45, 30]


@pytest.mark.parametrize(
    ("status", "code"),
    [
        (400, "bad_request"),
        (401, "bad_key"),
        (403, "forbidden"),
        (404, "not_found"),
        (429, "insufficient_quota"),
        (429, "billing_hard_limit_reached"),
    ],
)
def test_permanent_errors_are_not_retried(status, code, delays):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(status, json={"error": {"code": code}})

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(
                httpx.MockTransport(respond),
                max_attempts=None,
                retry_budget_seconds=None,
            )
        ) as client:
            result = await client.post("https://model.invalid")
            assert result.status_code == status
            assert result.json()["error"]["code"] == code

    asyncio.run(run())
    assert len(calls) == 1
    assert delays == []


def test_exhausted_retry_preserves_sdk_rate_limit_error_and_body(delays):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"code": "1302", "message": "busy"}})

    async def run():
        async with AsyncOpenAI(
            api_key="test",
            base_url="https://model.invalid",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=ModelRetryTransport(
                    httpx.MockTransport(respond), max_attempts=3
                )
            ),
        ) as client:
            with pytest.raises(RateLimitError) as error:
                await client.chat.completions.create(
                    model="test", messages=[{"role": "user", "content": "test"}]
                )
            assert error.value.status_code == 429
            assert error.value.body["code"] == "1302"

    asyncio.run(run())
    assert len(calls) == 3
    assert delays == [15, 30]


def test_server_delay_beyond_budget_stops_without_retrying_early(delays):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(
            429, headers={"Retry-After": "600"}, json={"error": "busy"}
        )

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(httpx.MockTransport(respond))
        ) as client:
            assert (await client.post("https://model.invalid")).status_code == 429

    asyncio.run(run())
    assert len(calls) == 1
    assert delays == []


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
def test_connection_errors_recover_with_backoff(failure, delays):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise failure("temporary network failure", request=request)
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(httpx.MockTransport(respond))
        ) as client:
            assert (await client.post("https://model.invalid")).status_code == 200

    asyncio.run(run())
    assert len(calls) == 2
    assert delays == [15]


def test_cancellation_is_not_retried(delays):
    def respond(request):
        raise asyncio.CancelledError

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(httpx.MockTransport(respond))
        ) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post("https://model.invalid")

    asyncio.run(run())
    assert delays == []


def test_non_streaming_response_body_failure_is_retried(delays):
    calls = []
    closed = []

    class BrokenBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"partial":'
            raise httpx.ReadError("connection lost while reading body")

        async def aclose(self):
            closed.append(True)

    def respond(request):
        calls.append(request.content)
        if len(calls) == 1:
            return httpx.Response(
                200, headers={"content-type": "application/json"}, stream=BrokenBody()
            )
        return httpx.Response(200, json={"complete": True})

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(httpx.MockTransport(respond))
        ) as client:
            response = await client.post("https://model.invalid", json={"x": 1})
            assert response.json() == {"complete": True}

    asyncio.run(run())
    assert len(calls) == 2 and calls[0] == calls[1]
    assert closed == [True]
    assert delays == [15]


def test_unlimited_retries_outlast_old_budget_then_recover(delays):
    bodies = []

    def respond(request):
        bodies.append(request.content)
        if len(bodies) <= 9:
            return httpx.Response(
                429,
                headers={"Retry-After": "600"} if len(bodies) == 8 else {},
                json={"error": {"code": "1302"}},
            )
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(
                httpx.MockTransport(respond),
                max_attempts=None,
                retry_budget_seconds=None,
                pacer=RequestPacer(5),
            )
        ) as client:
            assert (
                await client.post("https://model.invalid", json={"x": 1})
            ).is_success

    asyncio.run(run())
    assert len(bodies) == 10 and len(set(bodies)) == 1
    assert delays == [15, 30, 60, 120, 240, 300, 300, 600, 300]


def test_cancellation_during_unlimited_backoff_stops_requests(monkeypatch, delays):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={})

    async def cancel_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr(retries.asyncio, "sleep", cancel_sleep)

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(
                httpx.MockTransport(respond),
                max_attempts=None,
                retry_budget_seconds=None,
            )
        ) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.post("https://model.invalid")

    asyncio.run(run())
    assert len(calls) == 1


def test_pacing_and_cooldown_are_shared_across_clients_and_event_loops(delays):
    pacer = RequestPacer(5)
    starts = []

    def respond(request):
        starts.append(retries.time.monotonic())
        return httpx.Response(429 if len(starts) == 1 else 200, json={})

    async def request_once():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(
                httpx.MockTransport(respond), max_attempts=1, pacer=pacer
            )
        ) as client:
            return await client.post("https://model.invalid")

    assert asyncio.run(request_once()).status_code == 429
    assert asyncio.run(request_once()).is_success
    assert asyncio.run(request_once()).is_success
    assert starts == [0, 15, 20]


def test_waiting_request_observes_extended_cooldown(monkeypatch, delays):
    pacer = RequestPacer(5)
    advance_time = retries.asyncio.sleep

    async def sleep(seconds):
        await advance_time(seconds)
        if len(delays) == 1:
            pacer.defer(20)

    monkeypatch.setattr(retries.asyncio, "sleep", sleep)
    asyncio.run(pacer.wait())
    asyncio.run(pacer.wait())
    assert delays == [5, 20]


def test_concurrent_requests_are_spaced():
    starts = []

    def respond(request):
        starts.append(retries.time.monotonic())
        return httpx.Response(200)

    async def run():
        async with httpx.AsyncClient(
            transport=ModelRetryTransport(
                httpx.MockTransport(respond), pacer=RequestPacer(0.02)
            )
        ) as client:
            await asyncio.gather(
                *(client.get("https://model.invalid") for _ in range(4))
            )

    asyncio.run(run())
    assert len(starts) == 4
    assert all(b - a >= 0.019 for a, b in pairwise(starts))


def test_retry_after_parsing():
    date = format_datetime(datetime.now(UTC) + timedelta(seconds=90))
    assert (
        88
        < retry_after_seconds(httpx.Response(429, headers={"Retry-After": date}))
        <= 90
    )
    for value in ("bad", "-1", "nan", "inf"):
        assert (
            retry_after_seconds(httpx.Response(429, headers={"Retry-After": value}))
            is None
        )
    assert (
        retry_after_seconds(httpx.Response(429, headers={"Retry-After": "20.5"}))
        == 20.5
    )


@pytest.mark.parametrize(
    "model_class,provider_class",
    [(OpenAIChatModel, OpenAIProvider), (ZaiModel, ZaiProvider)],
)
def test_retry_after_tool_call_does_not_restart_agent_work(
    delays, model_class, provider_class
):
    requests = []
    tool_calls = []

    def respond(request):
        body = json.loads(request.content)
        requests.append(body)
        if len(requests) == 1:
            message = {
                "role": "assistant",
                "content": None,
                "reasoning_content": "Need to look up the answer first.",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "lookup", "arguments": "{}"},
                    }
                ],
            }
            finish = "tool_calls"
        elif len(requests) == 2:
            return httpx.Response(
                429, json={"error": {"code": "1302", "message": "busy"}}
            )
        else:
            message = {"role": "assistant", "content": "42"}
            finish = "stop"
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "test",
                "choices": [{"index": 0, "message": message, "finish_reason": finish}],
                "usage": {
                    "prompt_tokens": 1,
                    "completion_tokens": 1,
                    "total_tokens": 2,
                },
            },
        )

    async def run():
        async with AsyncOpenAI(
            api_key="test",
            base_url="https://model.invalid",
            max_retries=0,
            http_client=httpx.AsyncClient(
                transport=ModelRetryTransport(httpx.MockTransport(respond))
            ),
        ) as client:
            agent = Agent(
                model_class("glm-5.3", provider=provider_class(openai_client=client))
            )

            @agent.tool_plain
            def lookup() -> int:
                tool_calls.append(True)
                return 42

            assert (await agent.run("Look up the answer")).output == "42"

    asyncio.run(run())
    assert tool_calls == [True]
    assert len(requests) == 3
    assert requests[1] == requests[2]
    assistant = next(m for m in requests[2]["messages"] if m["role"] == "assistant")
    assert assistant["reasoning_content"] == "Need to look up the answer first."
    if model_class is ZaiModel:
        assert all(r["thinking"] == {"clear_thinking": False} for r in requests)
    assert delays == [15]


@pytest.mark.parametrize(
    "provider_class", [RetryingOpenAIProvider, RetryingZaiProvider]
)
def test_provider_closes_and_recreates_client_without_sdk_retries(provider_class):
    async def run():
        provider = provider_class(api_key="test")
        async with provider:
            first = provider._own_http_client
            assert not first.is_closed
            assert provider.client.max_retries == 0
        assert first.is_closed
        async with provider:
            assert provider._own_http_client is not first
            assert not provider._own_http_client.is_closed
            assert provider.client.max_retries == 0
        assert provider._own_http_client.is_closed

    asyncio.run(run())


def test_model_retry_settings_are_recorded_in_metadata(monkeypatch):
    monkeypatch.setattr(
        AgentConfig,
        "load_provider",
        lambda _: OpenAIProviderConfig(_env_file=None, api_key="test"),
    )
    config = AgentConfig(
        _env_file=None,
        catalog_url="http://catalog.invalid",
        catalog_token="test",
        spark_url="sc://spark.invalid",
        otel_exporter_otlp_endpoint="http://otel.invalid",
        model_provider="openai",
        model_request_max_attempts=4,
        model_request_retry_budget_seconds=120,
        model_request_min_interval_seconds=5,
    )
    metadata = config.experiment_metadata()
    assert metadata["model_request_max_attempts"] == 4
    assert metadata["model_request_retry_budget_seconds"] == 120
    assert metadata["model_request_min_interval_seconds"] == 5

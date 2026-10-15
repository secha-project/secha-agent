import asyncio
import json
from contextlib import asynccontextmanager

import httpx
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.providers.zai import ZaiProvider

from src.agent.config import AgentConfig, OpenAIProviderConfig, create_provider


@asynccontextmanager
async def sdk_client(respond, *, max_attempts=6, zai=False):
    async with create_provider(
        api_key="test",
        base_url="https://model.invalid",
        max_attempts=max_attempts,
        zai=zai,
    ) as provider:
        await provider.client._client._transport.aclose()
        provider.client._client._transport = httpx.MockTransport(respond)
        yield provider.client


@pytest.fixture
def delays(monkeypatch):
    recorded = []

    async def sleep(seconds):
        recorded.append(seconds)

    monkeypatch.setattr("openai._base_client.anyio.sleep", sleep)
    monkeypatch.setattr("openai._base_client.random", lambda: 0)
    return recorded


@pytest.mark.parametrize("status", [408, 409, 429, 500, 501, 502, 503, 504])
def test_sdk_transient_errors_retry_the_same_request(status, delays):
    bodies = []

    def respond(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(status, headers={"Retry-After": "45"}, json={})
        if len(bodies) == 2:
            return httpx.Response(status, json={})
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with sdk_client(respond) as client:
            result = await client.post(
                "/", cast_to=httpx.Response, body={"prompt": "test"}
            )
            assert result.json() == {"ok": True}

    asyncio.run(run())
    assert len(bodies) == 3 and len(set(bodies)) == 1
    assert delays == [45, 1]


@pytest.mark.parametrize(
    "status, body, expected_calls",
    [
        (400, b'{"detail":"Open WebUI: Server Connection Error"}', 2),
        (400, b'{"detail":"Open WebUI: Invalid model"}', 1),
        (400, b'{"detail":"Server Connection Error"}', 1),
        (400, b'["Open WebUI: Server Connection Error"]', 1),
        (400, b"not JSON", 1),
        (401, b'{"error":{"code":"bad_key"}}', 1),
        (403, b'{"error":{"code":"forbidden"}}', 1),
        (404, b'{"error":{"code":"not_found"}}', 1),
        (429, b'{"error":{"code":"insufficient_quota"}}', 1),
        (429, b'{"error":{"code":"billing_hard_limit_reached"}}', 1),
        (429, b'{"error":{"code":"1302"}}', 2),
        (429, b"not JSON", 2),
    ],
)
def test_tuni_and_quota_exceptions_with_unread_response_body(
    status, body, expected_calls, delays
):
    bodies = []
    closed = []

    class ErrorBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield body[:5]
            yield body[5:]

        async def aclose(self):
            closed.append(True)

    def respond(request):
        bodies.append(request.content)
        if len(bodies) == 1:
            return httpx.Response(status, stream=ErrorBody())
        assert closed == [True]
        return httpx.Response(200, json={"ok": True})

    async def run():
        async with sdk_client(respond, max_attempts=2) as client:
            if expected_calls == 1:
                with pytest.raises(APIStatusError) as caught:
                    await client.post(
                        "/", cast_to=httpx.Response, body={"prompt": "test"}
                    )
                assert caught.value.status_code == status
                assert caught.value.response.content == body
            else:
                response = await client.post(
                    "/", cast_to=httpx.Response, body={"prompt": "test"}
                )
                assert response.is_success

    asyncio.run(run())
    assert len(bodies) == expected_calls and len(set(bodies)) == 1
    assert closed == [True]
    assert delays == ([0.5] if expected_calls == 2 else [])


@pytest.mark.parametrize("max_attempts", [1, 3, 6])
def test_exhaustion_preserves_sdk_rate_limit_error_and_body(max_attempts, delays):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={"error": {"code": "1302", "message": "busy"}})

    async def run():
        async with sdk_client(respond, max_attempts=max_attempts) as client:
            with pytest.raises(RateLimitError) as caught:
                await client.chat.completions.create(model="test", messages=[])
            assert caught.value.status_code == 429
            assert caught.value.body["code"] == "1302"

    asyncio.run(run())
    assert len(calls) == max_attempts
    assert len(delays) == max_attempts - 1


@pytest.mark.parametrize(
    "header, expected_delay", [("2", 2), ("20.5", 20.5), ("600", 0.5), ("invalid", 0.5)]
)
def test_sdk_owns_retry_after_handling(header, expected_delay, delays):
    async def run():
        async with sdk_client(
            lambda _: httpx.Response(503, headers={"Retry-After": header}),
            max_attempts=2,
        ) as client:
            with pytest.raises(APIStatusError):
                await client.get("/", cast_to=httpx.Response)

    asyncio.run(run())
    assert delays == [expected_delay]


@pytest.mark.parametrize("failure", [httpx.ConnectError, httpx.ReadTimeout])
def test_sdk_connection_errors_recover_with_backoff(failure, delays):
    calls = []

    def respond(request):
        calls.append(request)
        if len(calls) == 1:
            raise failure("temporary network failure", request=request)
        return httpx.Response(200)

    async def run():
        async with sdk_client(respond) as client:
            assert (await client.get("/", cast_to=httpx.Response)).status_code == 200

    asyncio.run(run())
    assert len(calls) == 2
    assert delays == [0.5]


@pytest.mark.parametrize(
    "failure, sdk_error",
    [(httpx.ConnectError, APIConnectionError), (httpx.ReadTimeout, APITimeoutError)],
)
def test_sdk_exhaustion_preserves_network_error_as_cause(failure, sdk_error, delays):
    error = failure("request failed")
    calls = []

    def respond(request):
        calls.append(request)
        raise error

    async def run():
        async with sdk_client(respond, max_attempts=2) as client:
            with pytest.raises(sdk_error) as caught:
                await client.get("/", cast_to=httpx.Response)
            assert caught.value.__cause__ is error

    asyncio.run(run())
    assert len(calls) == 2
    assert delays == [0.5]


def test_sdk_recovers_from_non_streaming_body_read_failure(delays):
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
            return httpx.Response(200, stream=BrokenBody())
        return httpx.Response(200, json={"complete": True})

    async def run():
        async with sdk_client(respond, max_attempts=2) as client:
            response = await client.post("/", cast_to=httpx.Response, body={"x": 1})
            assert response.json() == {"complete": True}

    asyncio.run(run())
    assert len(calls) == 2 and calls[0] == calls[1]
    assert closed == [True]
    assert delays == [0.5]


def test_partial_sse_stream_is_not_replayed(delays):
    calls = []
    closed = []

    class BrokenStream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'data: {"id":"chunk","choices":[{"index":0,"delta":{"content":"partial"}}]}\n\n'
            raise httpx.ReadError("stream disconnected")

        async def aclose(self):
            closed.append(True)

    def respond(request):
        calls.append(request)
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"}, stream=BrokenStream()
        )

    async def run():
        async with (
            sdk_client(respond) as client,
            await client.chat.completions.create(
                model="test", messages=[], stream=True
            ) as stream,
        ):
            chunks = stream.__aiter__()
            assert (await anext(chunks)).choices[0].delta.content == "partial"
            with pytest.raises(httpx.ReadError, match="stream disconnected"):
                await anext(chunks)

    asyncio.run(run())
    assert len(calls) == 1
    assert closed == [True]
    assert delays == []


@pytest.mark.parametrize("status", [200, 400, 429, 503])
def test_cancellation_during_body_read_closes_response(status, delays):
    calls = []
    closed = []

    class CancelledBody(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b'{"partial":'
            raise asyncio.CancelledError

        async def aclose(self):
            closed.append(True)

    def respond(request):
        calls.append(request)
        return httpx.Response(status, stream=CancelledBody())

    async def run():
        async with sdk_client(respond) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.get("/", cast_to=httpx.Response)

    asyncio.run(run())
    assert len(calls) == 1
    assert closed == [True]
    assert delays == []


def test_cancellation_during_backoff_stops_requests(monkeypatch):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(429, json={})

    async def cancel_sleep(_seconds):
        raise asyncio.CancelledError

    monkeypatch.setattr("openai._base_client.anyio.sleep", cancel_sleep)

    async def run():
        async with sdk_client(respond) as client:
            with pytest.raises(asyncio.CancelledError):
                await client.get("/", cast_to=httpx.Response)

    asyncio.run(run())
    assert len(calls) == 1


@pytest.mark.parametrize("zai", [False, True])
def test_native_provider_recreation_preserves_hook_and_sdk_retries(zai, delays):
    async def run():
        provider = create_provider(api_key="test", zai=zai)
        assert type(provider) is (ZaiProvider if zai else OpenAIProvider)
        previous = None
        for _ in range(2):
            async with provider:
                http_client = provider.client._client
                assert http_client is not previous and not http_client.is_closed
                assert provider.client.max_retries == 5
                assert http_client.timeout.read == 600
                assert http_client.timeout.connect == 5
                await http_client._transport.aclose()
                http_client._transport = httpx.MockTransport(
                    lambda _: httpx.Response(
                        429,
                        headers={"x-should-retry": "true"},
                        json={"error": {"code": "insufficient_quota"}},
                    )
                )
                with pytest.raises(RateLimitError):
                    await provider.client.get("/", cast_to=httpx.Response)
            assert http_client.is_closed
            previous = http_client

    asyncio.run(run())
    assert delays == []


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
        async with sdk_client(respond) as client:
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
    assert delays == [0.5]


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
    )
    metadata = config.experiment_metadata()
    assert metadata["model_request_max_attempts"] == 4
    assert "model_request_retry_budget_seconds" not in metadata
    assert "model_request_min_interval_seconds" not in metadata

    default_config = AgentConfig(
        _env_file=None,
        catalog_url="http://catalog.invalid",
        catalog_token="test",
        spark_url="sc://spark.invalid",
        otel_exporter_otlp_endpoint="http://otel.invalid",
        model_provider="openai",
    )
    assert default_config.model_request_max_attempts == 6
    assert default_config.experiment_metadata()["model_request_max_attempts"] == 6

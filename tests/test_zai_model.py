import asyncio
import json

import httpx
import pytest
from pydantic_ai.models.zai import ZaiModel

from src.agent.agent import create_agent
from src.agent.config import AgentConfig, ZaiProviderConfig
from src.agent.model_retry import ModelRetryTransport, RetryingZaiProvider


@pytest.fixture
def config(monkeypatch):
    monkeypatch.setattr(
        AgentConfig,
        "load_provider",
        lambda _: ZaiProviderConfig(
            _env_file=None,
            api_key="test",
            url="https://zai.invalid/custom/",
            model_name="glm-5.3",
        ),
    )
    return AgentConfig(
        _env_file=None,
        catalog_url="http://catalog.invalid",
        catalog_token="test",
        spark_url="sc://spark.invalid",
        otel_exporter_otlp_endpoint="http://otel.invalid",
        model_provider="zai",
        model_name="glm-5.3",
        model_request_max_attempts=None,
        model_request_retry_budget_seconds=None,
        model_request_min_interval_seconds=5,
    )


def test_zai_uses_native_adapter_with_shared_retry_policy(config):
    async def run():
        model = config.create_model()
        assert isinstance(model, ZaiModel)
        assert isinstance(model.provider, RetryingZaiProvider)
        assert model.provider.name == "zai"
        assert model.provider.base_url == "https://zai.invalid/custom/"
        assert model.client.max_retries == 0
        async with model.provider:
            transport = model.client._client._transport
            assert isinstance(transport, ModelRetryTransport)
            assert transport.max_attempts is None
            assert transport.retry_budget_seconds is None
            assert transport.pacer.min_interval_seconds == 5

    asyncio.run(run())
    assert config.experiment_metadata()["provider_settings"]["model_adapter"] == (
        "zai_inline_defs_v1"
    )


def test_real_visualization_tools_send_inline_object_schemas(config):
    requests = []

    def respond(request):
        assert str(request.url) == "https://zai.invalid/custom/chat/completions"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "completion",
                "object": "chat.completion",
                "created": 0,
                "model": "glm-5.3",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "done"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async def run():
        model = config.create_model()
        async with model.provider:
            transport = model.client._client._transport
            await transport.wrapped.aclose()
            transport.wrapped = httpx.MockTransport(respond)
            result = await create_agent(config.enabled_capabilities).run(
                "question", model=model, deps=object()
            )
            assert result.output == "done"

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0]["thinking"] == {"clear_thinking": False}
    tools = {t["function"]["name"]: t["function"] for t in requests[0]["tools"]}
    for name in (
        "inspect_table_output",
        "display_table_output",
        "visualize_table_output",
    ):
        schema = tools[name]["parameters"]
        assert "$ref" not in json.dumps(schema)
        assert "$defs" not in json.dumps(schema)
        ref = schema["properties"]["ref"]
        assert ref["type"] == "object"
        assert ref["properties"]["output_id"]["type"] == "string"
        assert ref["required"] == ["output_id"]

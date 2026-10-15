import asyncio
import base64
import json
from types import SimpleNamespace
from typing import Any, cast

import httpx
import pandas as pd
import pytest
from pydantic_ai import BinaryContent, ModelMessagesTypeAdapter, UserPromptPart
from pydantic_ai.models.zai import ZaiModel
from pydantic_ai.providers.zai import ZaiProvider

from src.agent.agent import create_agent
from src.agent.config import AgentConfig, ZaiProviderConfig
from src.agent.deps import Deps
from src.agent.outputs.tables import LiveTableOutput


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
        model_request_max_attempts=6,
    )


def test_zai_uses_native_adapter_with_shared_retry_policy(config):
    async def run():
        model = config.create_model()
        assert isinstance(model, ZaiModel)
        assert isinstance(model.provider, ZaiProvider)
        assert model.provider.name == "zai"
        assert str(model.client.base_url) == "https://zai.invalid/custom/"
        assert model.client.max_retries == 5
        async with model.provider:
            transport = model.client._client._transport
            assert type(transport) is httpx.AsyncHTTPTransport

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
            await transport.aclose()
            model.client._client._transport = httpx.MockTransport(respond)
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
        "display_chart_output",
    ):
        schema = tools[name]["parameters"]
        assert "$ref" not in json.dumps(schema)
        assert "$defs" not in json.dumps(schema)
        ref = schema["properties"]["ref"]
        assert ref["type"] == "object"
        assert ref["properties"]["output_id"]["type"] == "string"
        assert ref["required"] == ["output_id"]


def test_zai_agent_receives_png_revises_and_publishes_only_selected_chart(config):
    deps = Deps(
        spark=cast(Any, object()),
        catalog=cast(Any, object()),
        schema_name="bronze",
        table_column_page_size=20,
    )

    async def sample(_max_rows):
        return pd.DataFrame({"site": ["East", "West"], "power": [12, 27]}), False

    table_ref = deps.table_outputs.add(
        cast(LiveTableOutput, SimpleNamespace(sample=sample))
    )
    requests = []
    chart_refs = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        step = len(requests)
        assert payload["model"] == "glm-5.3-flash"
        for item in payload["messages"]:
            if (
                item["role"] == "tool"
                and item["tool_call_id"] == f"call-{step - 1}"
                and step <= 3
            ):
                chart_refs.append(json.loads(item["content"])["output_ref"])
        message = {"role": "assistant", "content": None}
        if step <= 2:
            assert deps.artifacts.published == ()
            name = "visualize_table_output"
            args = {
                "ref": table_ref.model_dump(),
                "ggsql": (
                    "VISUALISE site AS x, power AS y DRAW bar "
                    f"LABEL x => 'Site', y => 'Power ({'W' if step == 1 else 'kW'})'"
                ),
                "title": "Power by site",
            }
        elif step == 3:
            assert deps.artifacts.published == ()
            name = "display_chart_output"
            args = {"ref": chart_refs[-1]}
        else:
            assert step == 4
            assert len(deps.artifacts.published) == 1
            message["content"] = "The corrected chart is available."

        if step < 4:
            message["tool_calls"] = [
                {
                    "id": f"call-{step}",
                    "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)},
                }
            ]
        message["reasoning_content"] = f"Review step {step}."
        return httpx.Response(
            200,
            json={
                "id": f"completion-{step}",
                "object": "chat.completion",
                "created": 0,
                "model": "glm-5.3-flash",
                "choices": [
                    {
                        "index": 0,
                        "message": message,
                        "finish_reason": "tool_calls" if step < 4 else "stop",
                    }
                ],
            },
        )

    async def run():
        model = config.model_copy(
            update={
                "model_name": "glm-5.3-flash",
            }
        ).create_model()
        async with model.provider:
            transport = model.client._client._transport
            await transport.aclose()
            model.client._client._transport = httpx.MockTransport(respond)
            return await create_agent(config.enabled_capabilities).run(
                "Plot power by site in kW, inspect it, and publish the corrected chart.",
                model=model,
                deps=deps,
            )

    result = asyncio.run(run())
    assert result.output == "The corrected chart is available."
    assert len(requests) == 4
    assert len(chart_refs) == 2
    assert chart_refs[0] != chart_refs[1]
    assert (
        deps.artifacts.published[0].spec["layer"][0]["encoding"]["y"]["title"]
        == "Power (kW)"
    )
    for step, expected_count in ((1, 1), (2, 2)):
        images = [
            part["image_url"]["url"]
            for message in requests[step]["messages"]
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        assert len(images) == expected_count
        assert all(
            uri.startswith("data:image/png;base64,")
            and base64.b64decode(uri.split(",", 1)[1]).startswith(b"\x89PNG\r\n\x1a\n")
            for uri in images
        )

    # Image content must survive the app's conversation-history serialization.
    restored = ModelMessagesTypeAdapter.validate_json(result.all_messages_json())
    images = [
        item
        for message in restored
        for part in message.parts
        if isinstance(part, UserPromptPart) and isinstance(part.content, list)
        for item in part.content
        if isinstance(item, BinaryContent)
    ]
    assert len(images) == 2

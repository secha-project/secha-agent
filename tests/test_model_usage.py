"""Exercise real provider parsing; no model API or database requests."""

import asyncio
from types import SimpleNamespace
from uuid import uuid4

import httpx
import logfire
import pytest
from openai import AsyncOpenAI
from pydantic_ai import Agent
from pydantic_ai.models.instrumented import InstrumentationSettings, InstrumentedModel
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider
from pydantic_ai.usage import RequestUsage
from pydantic_evals import Case, Dataset

from src.validation.evaluators.report import TokenUsageAndCostEvaluator


@pytest.mark.parametrize(
    ("provider", "model", "flavor", "raw"),
    [
        (
            "openai",
            "gpt-6-sol",
            "responses",
            {
                "input_tokens": 1849,
                "output_tokens": 24,
                "total_tokens": 1873,
                "input_tokens_details": {
                    "cached_tokens": 0,
                    "cache_write_tokens": 1846,
                },
                "output_tokens_details": {"reasoning_tokens": 7},
            },
        ),
        (
            "zai",
            "glm-5.3",
            "chat",
            {
                "prompt_tokens": 1849,
                "completion_tokens": 24,
                "total_tokens": 1873,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        ),
        (
            "openai",
            "Qwen3.8-27B-AIC",
            "chat",
            {
                "prompt_tokens": 1849,
                "completion_tokens": 24,
                "total_tokens": 1873,
                "completion_tokens_details": {"reasoning_tokens": 7},
            },
        ),
    ],
)
def test_reasoning_usage_does_not_silently_erase_token_totals(
    provider, model, flavor, raw
):
    usage = RequestUsage.extract(
        {"model": model, "usage": raw},
        provider=provider,
        provider_url="https://unused.invalid/v1",
        provider_fallback="openai",
        api_flavor=flavor,
    )
    assert usage.input_tokens == 1849
    assert usage.output_tokens == 24
    assert usage.total_tokens == 1873  # Reasoning is already part of output.
    if flavor == "responses":
        assert usage.cache_write_tokens == 1846


def test_responses_usage_reaches_benchmark_report_once(monkeypatch):
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT", raising=False)
    logfire.configure(send_to_logfire=False, console=False)
    calls = []

    def respond(request):
        calls.append(request)
        first = len(calls) == 1
        return httpx.Response(
            200,
            json={
                "id": f"resp_{len(calls)}",
                "object": "response",
                "created_at": 0,
                "status": "completed",
                "model": "gpt-6-sol",
                "parallel_tool_calls": True,
                "tool_choice": "auto",
                "tools": [],
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_test",
                        "call_id": "call_test",
                        "name": "lookup",
                        "arguments": "{}",
                    }
                ]
                if first
                else [
                    {
                        "type": "message",
                        "id": "msg_test",
                        "role": "assistant",
                        "status": "completed",
                        "content": [
                            {"type": "output_text", "text": "done", "annotations": []}
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 1849 if first else 2200,
                    "input_tokens_details": {
                        "cached_tokens": 0 if first else 1846,
                        "cache_write_tokens": 1846 if first else 351,
                    },
                    "output_tokens": 24 if first else 122,
                    "output_tokens_details": {"reasoning_tokens": 7 if first else 13},
                    "total_tokens": 1873 if first else 2322,
                },
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            model = OpenAIResponsesModel(
                "gpt-6-sol",
                provider=OpenAIProvider(
                    openai_client=AsyncOpenAI(api_key="test", http_client=http)
                ),
            )
            agent = Agent(InstrumentedModel(model, InstrumentationSettings()))

            @agent.tool_plain
            def lookup() -> str:
                return "data"

            async def task(prompt):
                return (await agent.run(prompt)).output

            return await Dataset(
                name="usage-regression",
                cases=[Case(name=str(uuid4()), inputs="question")],
            ).evaluate(task)

    report = asyncio.run(run())
    assert len(calls) == 2
    assert not report.failures
    metrics = report.cases[0].metrics
    assert metrics["requests"] == 2
    assert metrics["input_tokens"] == 4049
    assert metrics["output_tokens"] == 146
    assert metrics["cache_read_tokens"] == 1846
    assert metrics["cache_write_tokens"] == 2197
    assert metrics["reasoning_tokens"] == 20
    report.experiment_metadata = {
        "agent": {
            "model_provider": "openai",
            "provider_settings": {"model_name": "gpt-6-sol"},
        }
    }
    summary = TokenUsageAndCostEvaluator().evaluate(SimpleNamespace(report=report))
    values = {row[0]: row[1] for row in summary.rows}
    assert values["Total tokens"] == 4195
    assert values["Estimated cost"] == pytest.approx(0.0073337)

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic_ai import Agent
from pydantic_ai.messages import ModelRequest, UserPromptPart
from pyspark.sql import SparkSession

from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.outputs.tables import TableOutputStore
from src.agent.service import AgentRun, AgentService


class AgentStub:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    @asynccontextmanager
    async def run_stream_events(
        self,
        prompt: str,
        **kwargs: Any,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        self.calls.append((prompt, kwargs))

        async def events() -> AsyncIterator[Any]:
            yield "started"
            yield "completed"

        yield events()


def agent_service(
    agent: AgentStub, *, temperature: float | None, semantic: bool
) -> AgentService:
    config = cast(
        AgentConfig,
        SimpleNamespace(
            schema_name="test_schema",
            table_column_page_size=7,
            temperature=temperature,
            request_limit=4,
            enabled_capabilities=("catalog", "query", "table_outputs")
            + (("semantic_layer",) if semantic else ()),
            create_model=lambda: "test-model",
        ),
    )
    return AgentService(
        config=config,
        agent=cast(Agent[Deps, str], agent),
        spark=cast(SparkSession, object()),
        catalog=cast(UnityCatalogClient, object()),
    )


def collect(run: AgentRun) -> list[Any]:
    async def collect_events() -> list[Any]:
        return [event async for event in run.events]

    return asyncio.run(collect_events())


@pytest.mark.parametrize("temperature, semantic", [(0.25, True), (None, False)])
def test_agent_service_passes_configuration_and_isolates_run_artifacts(
    temperature: float | None, semantic: bool
) -> None:
    agent = AgentStub()
    service = agent_service(agent, temperature=temperature, semantic=semantic)
    table_outputs = TableOutputStore()
    histories = [[], [ModelRequest(parts=[UserPromptPart(content="earlier question")])]]

    first = service.start_run(
        prompt="question",
        message_history=histories[0],
        table_outputs=table_outputs,
    )
    second = service.start_run(
        prompt="follow-up",
        message_history=histories[1],
        table_outputs=table_outputs,
    )

    assert collect(first) == ["started", "completed"]
    assert collect(second) == ["started", "completed"]
    assert first.artifacts is not second.artifacts
    assert [prompt for prompt, _ in agent.calls] == ["question", "follow-up"]
    for (_, kwargs), history, run in zip(
        agent.calls, histories, (first, second), strict=True
    ):
        assert kwargs["message_history"] is history
        assert kwargs["model"] == "test-model"
        assert kwargs["model_settings"] == (
            {"temperature": temperature} if temperature is not None else {}
        )
        assert kwargs["usage_limits"].request_limit == 4
        deps = kwargs["deps"]
        assert deps.spark is service.spark
        assert deps.catalog is service.catalog
        assert deps.schema_name == "test_schema"
        assert deps.table_column_page_size == 7
        assert deps.semantic_layer_enabled is semantic
        assert deps.table_outputs is table_outputs
        assert deps.artifacts is run.artifacts

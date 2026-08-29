import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, cast

from pydantic_ai import Agent
from pyspark.sql import SparkSession

from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.outputs.tables import TableOutputStore
from src.agent.service import AgentRun, AgentService


class AgentStub:
    @asynccontextmanager
    async def run_stream_events(
        self,
        _prompt: str,
        **_kwargs: Any,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        async def events() -> AsyncIterator[Any]:
            yield "started"
            yield "completed"

        yield events()


def agent_service(agent: AgentStub) -> AgentService:
    config = cast(
        AgentConfig,
        SimpleNamespace(
            schema_name="bronze",
            table_column_page_size=20,
            temperature=0.25,
            request_limit=4,
            create_model=lambda: "test-model",
        ),
    )
    return AgentService(
        config=config,
        agent=cast(Agent[Deps, str], agent),
        spark=cast(SparkSession, object()),
        catalog=cast(UnityCatalogClient, object()),
        semantic=None,
    )


def collect(run: AgentRun) -> list[Any]:
    async def collect_events() -> list[Any]:
        return [event async for event in run.events]

    return asyncio.run(collect_events())


def test_agent_service_owns_run_configuration_and_dependencies() -> None:
    agent = AgentStub()
    service = agent_service(agent)
    table_outputs = TableOutputStore()

    first = service.start_run(
        prompt="question",
        message_history=[],
        table_outputs=table_outputs,
    )
    second = service.start_run(
        prompt="follow-up",
        message_history=[],
        table_outputs=table_outputs,
    )

    assert collect(first) == ["started", "completed"]
    assert collect(second) == ["started", "completed"]
    assert first.artifacts is not second.artifacts

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass

from pydantic_ai import (
    Agent,
    AgentStreamEvent,
    ModelMessage,
    ModelSettings,
    UsageLimits,
)
from pydantic_ai.run import AgentRunResultEvent
from pyspark.sql import SparkSession

from .catalog import UnityCatalogClient
from .config import AgentConfig
from .deps import Deps
from .outputs.artifacts import ArtifactCollector
from .outputs.tables import TableOutputStore
from .semantic_runtime import SemanticRuntime

AgentEvent = AgentStreamEvent | AgentRunResultEvent[str]


@dataclass(frozen=True)
class AgentRun:
    events: AsyncIterator[AgentEvent]
    artifacts: ArtifactCollector


@dataclass(frozen=True)
class AgentService:
    """Configured agent runtime exposed to application frontends."""

    config: AgentConfig
    agent: Agent[Deps, str]
    spark: SparkSession
    catalog: UnityCatalogClient
    semantic: SemanticRuntime | None

    def start_run(
        self,
        *,
        prompt: str,
        message_history: list[ModelMessage],
        table_outputs: TableOutputStore,
    ) -> AgentRun:
        artifacts = ArtifactCollector()
        deps = Deps(
            spark=self.spark,
            catalog=self.catalog,
            schema_name=self.config.schema_name,
            table_column_page_size=self.config.table_column_page_size,
            semantic=self.semantic,
            table_outputs=table_outputs,
            artifacts=artifacts,
        )
        return AgentRun(
            events=self._stream_events(
                prompt=prompt,
                message_history=message_history,
                deps=deps,
            ),
            artifacts=artifacts,
        )

    async def _stream_events(
        self,
        *,
        prompt: str,
        message_history: list[ModelMessage],
        deps: Deps,
    ) -> AsyncIterator[AgentEvent]:
        async with self.agent.run_stream_events(
            prompt,
            deps=deps,
            model=self.config.create_model(),
            model_settings=ModelSettings(temperature=self.config.temperature),
            usage_limits=UsageLimits(request_limit=self.config.request_limit),
            message_history=message_history,
        ) as events:
            async for event in events:
                yield event

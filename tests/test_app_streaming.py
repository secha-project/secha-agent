import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from pydantic_ai import (
    AgentRunResult,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ToolCallPart,
    ToolReturnPart,
)

from src.agent.outputs.artifacts import ArtifactCollector, ArtifactRef, ChartArtifact
from src.app.models import ArtifactDisplayPart, ToolDisplayPart
from src.app.streaming import stream_agent_response


class StatusContainer:
    def container(self, **_kwargs: Any) -> "StatusContainer":
        return self

    def caption(self, _value: str) -> None:
        pass

    def markdown(self, _value: str) -> None:
        pass

    def code(self, _value: str, **_kwargs: Any) -> None:
        pass

    def update(self, **_kwargs: Any) -> None:
        pass


class StreamingContainer:
    def __init__(self) -> None:
        self.status_container = StatusContainer()
        self.chart_count = 0

    def status(self, *_args: Any, **_kwargs: Any) -> StatusContainer:
        return self.status_container

    def container(self, **_kwargs: Any) -> "StreamingContainer":
        return self

    def columns(self, *_args: Any, **_kwargs: Any) -> list["StreamingContainer"]:
        return [self, self]

    def dataframe(self, _value: Any, **_kwargs: Any) -> None:
        pass

    def caption(self, _value: str) -> None:
        pass

    def download_button(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def vega_lite_chart(self, *_args: Any, **_kwargs: Any) -> None:
        self.chart_count += 1


class AgentStub:
    def __init__(self, artifact_ref: ArtifactRef) -> None:
        self.artifact_ref = artifact_ref

    @asynccontextmanager
    async def run_stream_events(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        async def events() -> AsyncIterator[Any]:
            yield FunctionToolCallEvent(
                ToolCallPart(
                    tool_name="display_table_output",
                    args={"ref": "df-1"},
                    tool_call_id="call-1",
                )
            )
            yield FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="display_table_output",
                    content="displayed",
                    tool_call_id="call-1",
                    metadata=self.artifact_ref,
                )
            )
            yield AgentRunResultEvent(AgentRunResult(output="done"))

        yield events()


async def agent_events(agent: Any) -> AsyncIterator[object]:
    async with agent.run_stream_events() as events:
        async for event in events:
            yield event


def test_stream_returns_owned_assistant_and_pending_artifacts(
    make_table_artifact: Any,
) -> None:
    artifacts = ArtifactCollector()
    artifact = make_table_artifact([{"id": 1}])
    artifact_ref = artifacts.publish(artifact)
    container = StreamingContainer()

    result = asyncio.run(
        stream_agent_response(
            events=agent_events(AgentStub(artifact_ref)),
            artifacts=artifacts,
            container=container,
        )
    )

    assert result is not None
    assert len(result.artifacts) == 1
    assert isinstance(result.assistant_message.parts[0], ToolDisplayPart)
    artifact_part = result.assistant_message.parts[1]
    assert isinstance(artifact_part, ArtifactDisplayPart)
    assert artifact_part.artifact_id == result.artifacts[0].id


class VisualizationAgentStub:
    def __init__(self, references: list[ArtifactRef]) -> None:
        self.references = references

    @asynccontextmanager
    async def run_stream_events(
        self,
        *_args: Any,
        **_kwargs: Any,
    ) -> AsyncIterator[AsyncIterator[Any]]:
        async def events() -> AsyncIterator[Any]:
            for index, reference in enumerate(self.references, start=1):
                tool_id = f"viz-{index}"
                yield FunctionToolCallEvent(
                    ToolCallPart(
                        tool_name="visualize_table_output",
                        args={"ggsql": "SELECT ...", "title": f"Attempt {index}"},
                        tool_call_id=tool_id,
                    )
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(
                        tool_name="visualize_table_output",
                        content="rendered",
                        tool_call_id=tool_id,
                        metadata=reference,
                    )
                )
            yield AgentRunResultEvent(AgentRunResult(output="done"))

        yield events()


def _make_chart_artifact(artifact_id: str) -> ChartArtifact:
    spec = {
        "data": {"name": "source"},
        "datasets": {"source": [{"x": 1, "y": 2}]},
        "mark": "point",
    }
    return ChartArtifact(id=artifact_id, spec=spec)


def test_stream_renders_each_published_visualization() -> None:
    run_artifacts = ArtifactCollector()
    published = [
        _make_chart_artifact("viz-first"),
        _make_chart_artifact("viz-last"),
    ]
    for artifact in published:
        run_artifacts.publish(artifact)
    container = StreamingContainer()

    result = asyncio.run(
        stream_agent_response(
            events=agent_events(
                VisualizationAgentStub([artifact.ref for artifact in published])
            ),
            artifacts=run_artifacts,
            container=container,
        )
    )

    assert result is not None
    parts = [
        part
        for part in result.assistant_message.parts
        if isinstance(part, ArtifactDisplayPart)
    ]
    assert len(parts) == 2
    assert container.chart_count == 2

import asyncio
import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from typing import Any, AsyncIterator, cast

from pydantic_ai import (
    AgentRunResult,
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelSettings,
    ToolCallPart,
    ToolReturnPart,
    UsageLimits,
)

from src.agent.artifacts.base import (
    Artifact,
    ArtifactContext,
    ArtifactRef,
)
from src.agent.artifacts.visualization import VisualizationContent
from src.agent.deps import Deps
from src.app.models import ArtifactDisplayPart, ToolDisplayPart
from src.app.streaming import stream_agent_response


class TaggedSpark:
    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


class StatusContainer:
    def __init__(self) -> None:
        self.container_kwargs: dict[str, Any] = {}

    def container(self, **kwargs: Any) -> "StatusContainer":
        self.container_kwargs = kwargs
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
        self.status_args: tuple[Any, ...] = ()
        self.status_kwargs: dict[str, Any] = {}
        self.status_container = StatusContainer()
        self.chart_count = 0

    def status(self, *args: Any, **kwargs: Any) -> StatusContainer:
        self.status_args = args
        self.status_kwargs = kwargs
        return self.status_container

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
                    tool_name="display_query_result",
                    args={"ref": "df-1"},
                    tool_call_id="call-1",
                )
            )
            yield FunctionToolResultEvent(
                ToolReturnPart(
                    tool_name="display_query_result",
                    content="displayed",
                    tool_call_id="call-1",
                    metadata=self.artifact_ref,
                )
            )
            yield AgentRunResultEvent(AgentRunResult(output="done"))

        yield events()


def test_stream_returns_owned_assistant_and_pending_artifacts(
    make_table_artifact: Any,
) -> None:
    artifact_context = ArtifactContext()
    artifact = make_table_artifact([{"id": 1}])
    artifact_context.registry.add(artifact)
    artifact_ref = asyncio.run(artifact_context.publish(artifact.ref))
    deps = cast(
        Deps,
        SimpleNamespace(
            spark=TaggedSpark(),
            artifact_context=artifact_context,
        ),
    )
    container = StreamingContainer()

    result = asyncio.run(
        stream_agent_response(
            prompt="show rows",
            message_history=[],
            deps=deps,
            agent_instance=AgentStub(artifact_ref),
            model=object(),
            model_settings=ModelSettings(),
            usage_limits=UsageLimits(),
            container=container,
        )
    )

    assert result is not None
    assert container.status_args == ("Used display query result",)
    assert container.status_kwargs["type"] == "compact"
    assert container.status_container.container_kwargs == {"gap": "xsmall"}
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
                        tool_name="visualize_query_result",
                        args={"ggsql": "SELECT ...", "title": f"Attempt {index}"},
                        tool_call_id=tool_id,
                    )
                )
                yield FunctionToolResultEvent(
                    ToolReturnPart(
                        tool_name="visualize_query_result",
                        content="rendered",
                        tool_call_id=tool_id,
                        metadata=reference,
                    )
                )
            yield AgentRunResultEvent(AgentRunResult(output="done"))

        yield events()


def _make_visualization_artifact(artifact_id: str) -> Artifact:
    spec = {
        "data": {"name": "source"},
        "datasets": {"source": [{"x": 1, "y": 2}]},
        "mark": "point",
    }
    return Artifact(
        ref=ArtifactRef(
            artifact_id=artifact_id,
            artifact_kind="visualization",
        ),
        content=VisualizationContent(spec_json=json.dumps(spec).encode()),
    )


def test_stream_renders_each_published_visualization() -> None:
    artifact_context = ArtifactContext()
    artifacts = [
        _make_visualization_artifact("viz-first"),
        _make_visualization_artifact("viz-last"),
    ]
    for artifact in artifacts:
        artifact_context.registry.add(artifact)
        asyncio.run(artifact_context.publish(artifact.ref))
    deps = cast(
        Deps,
        SimpleNamespace(
            spark=TaggedSpark(),
            artifact_context=artifact_context,
        ),
    )
    container = StreamingContainer()

    result = asyncio.run(
        stream_agent_response(
            prompt="make a chart",
            message_history=[],
            deps=deps,
            agent_instance=VisualizationAgentStub(
                [artifact.ref for artifact in artifacts]
            ),
            model=object(),
            model_settings=ModelSettings(),
            usage_limits=UsageLimits(),
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

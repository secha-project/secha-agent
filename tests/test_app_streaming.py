import asyncio
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

from src.agent.deps import Deps
from src.app.models import TableDisplayPart, ToolDisplayPart
from src.app.streaming import stream_agent_response


class MaterializedDataframe:
    def limit(self, _value: int) -> "MaterializedDataframe":
        return self

    def toPandas(self) -> list[dict[str, int]]:
        return [{"id": 1}]


class TaggedSpark:
    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


class ResultRegistryStub:
    def resolve(self, _ref: str) -> MaterializedDataframe:
        return MaterializedDataframe()


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


class AgentStub:
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
                    metadata={
                        "ui_kind": "dataframe",
                        "dataframe_ref": "df-1",
                    },
                )
            )
            yield AgentRunResultEvent(AgentRunResult(output="done"))

        yield events()


def test_stream_returns_owned_assistant_and_pending_artifacts() -> None:
    deps = cast(
        Deps,
        SimpleNamespace(
            spark=TaggedSpark(),
            result_registry=ResultRegistryStub(),
        ),
    )
    container = StreamingContainer()

    result = asyncio.run(
        stream_agent_response(
            prompt="show rows",
            message_history=[],
            deps=deps,
            agent_instance=AgentStub(),
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
    assert len(result.snapshots) == 1
    assert isinstance(result.assistant_message.parts[0], ToolDisplayPart)
    table_part = result.assistant_message.parts[1]
    assert isinstance(table_part, TableDisplayPart)
    assert table_part.artifact_id == result.snapshots[0].id

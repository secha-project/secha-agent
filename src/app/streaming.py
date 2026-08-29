from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
)

from src.agent.outputs.artifacts import Artifact, ArtifactCollector

from .models import AssistantDisplayMessage, TextDisplayPart, ToolDisplayPart
from .rendering import (
    ToolStatusView,
    finish_tool,
    format_json_string,
    json_for_display,
    render_tool,
    render_tool_result,
)


class ResponseEventRenderer:
    """Translate agent events into the live UI and persisted display message."""

    def __init__(self, container: Any, artifacts: ArtifactCollector) -> None:
        self.container = container
        self.artifacts = artifacts
        self.message = AssistantDisplayMessage()
        self._text_parts: dict[int, tuple[TextDisplayPart, Any]] = {}
        self._tool_parts: dict[str, tuple[ToolDisplayPart, ToolStatusView]] = {}

    def render(self, event: object) -> None:
        if isinstance(event, PartStartEvent) and isinstance(event.part, TextPart):
            part = TextDisplayPart(content=event.part.content)
            self.message.parts.append(part)
            placeholder = self.container.empty()
            placeholder.markdown(part.content)
            self._text_parts[event.index] = (part, placeholder)
            return

        if (
            isinstance(event, PartDeltaEvent)
            and isinstance(event.delta, TextPartDelta)
            and event.index in self._text_parts
        ):
            part, placeholder = self._text_parts[event.index]
            part.content += event.delta.content_delta
            placeholder.markdown(part.content)
            return

        if isinstance(event, PartEndEvent):
            self._text_parts.pop(event.index, None)
            return

        if isinstance(event, FunctionToolCallEvent):
            call = event.part
            part = ToolDisplayPart(
                tool_id=call.tool_call_id,
                title=call.tool_name.replace("_", " "),
                args=format_json_string(call.args_as_json_str()),
            )
            self.message.parts.append(part)
            self._tool_parts[call.tool_call_id] = (
                part,
                render_tool(self.container, part, live=True),
            )
            return

        if not isinstance(event, FunctionToolResultEvent):
            return
        tool = self._tool_parts.get(event.tool_call_id)
        if tool is None:
            return

        part, view = tool
        if isinstance(event.part, RetryPromptPart):
            finish_tool(
                view,
                part,
                output=json_for_display(event.part.content),
                state="error",
            )
            return

        try:
            display_part = render_tool_result(
                event.part.metadata,
                self.artifacts,
                self.container,
            )
            if display_part is not None:
                self.message.parts.append(display_part)
        except Exception as exc:
            finish_tool(
                view,
                part,
                output=f"Failed to display artifact: {exc}",
                state="error",
            )
            raise

        finish_tool(
            view,
            part,
            output=json_for_display(event.part.content),
            state="complete",
        )


@dataclass(frozen=True)
class StreamResult:
    assistant_message: AssistantDisplayMessage
    model_messages: list[ModelMessage]
    artifacts: tuple[Artifact, ...]


async def stream_agent_response(
    *,
    events: AsyncIterator[object],
    artifacts: ArtifactCollector,
    container: Any,
) -> StreamResult | None:
    """Render agent events and return display state after terminal success."""
    renderer = ResponseEventRenderer(container, artifacts)

    async for event in events:
        if isinstance(event, AgentRunResultEvent):
            return StreamResult(
                assistant_message=renderer.message,
                model_messages=event.result.all_messages(),
                artifacts=artifacts.published,
            )
        renderer.render(event)

    return None

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel
from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelSettings,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RetryPromptPart,
    TextPart,
    TextPartDelta,
    UsageLimits,
)

from src.agent.deps import Deps

from .dataframe_display import DataframeDisplayHandler
from .dataframes import DataframeArtifact
from .models import (
    AssistantDisplayMessage,
    TextDisplayPart,
    ToolDisplayPart,
)

TOOL_TO_DISPLAY_NAME = {
    "browse_tables": "Browse tables",
    "search_columns": "Search columns",
    "describe_table": "Describe table",
    "query_data": "Run query",
    "inspect_query_result": "Inspect query result",
    "display_query_result": "Display query result",
}


@dataclass(frozen=True)
class StreamResult:
    assistant_message: AssistantDisplayMessage
    model_messages: list[ModelMessage]
    artifacts: list[DataframeArtifact]


def json_for_display(value: Any) -> str:
    if isinstance(value, BaseModel):
        return value.model_dump_json(indent=2)
    try:
        return json.dumps(value, indent=2, ensure_ascii=False, default=str)
    except TypeError:
        return str(value)


def format_json_string(value: str) -> str:
    try:
        return json.dumps(json.loads(value), indent=2, ensure_ascii=False)
    except (json.JSONDecodeError, TypeError):
        return value


def render_tool(container: Any, part: ToolDisplayPart, *, live: bool = False) -> Any:
    status = container.status(
        part.title,
        expanded=live or part.state != "complete",
        state=part.state,
    )
    status.markdown("**Parameters**")
    status.code(part.args, language="json")
    if part.output is not None:
        heading = "Error" if part.state == "error" else "Output"
        status.markdown(f"**{heading}**")
        status.code(
            part.output,
            language="text" if part.state == "error" else "json",
        )
    return status


def _finish_tool(
    status: Any,
    part: ToolDisplayPart,
    *,
    output: str,
    state: Literal["complete", "error"],
) -> None:
    part.output = output
    part.state = state
    heading = "Error" if state == "error" else "Output"
    status.markdown(f"**{heading}**")
    status.code(output, language="text" if state == "error" else "json")
    status.update(
        label=part.title,
        state=state,
        expanded=state == "error",
    )


async def stream_agent_response(
    *,
    prompt: str,
    message_history: list[ModelMessage],
    deps: Deps,
    agent_instance: Any,
    model: Any,
    model_settings: ModelSettings,
    usage_limits: UsageLimits,
    container: Any,
) -> StreamResult | None:
    """Render ordered run events and return output only after terminal success."""
    assistant_message = AssistantDisplayMessage()
    dataframes = DataframeDisplayHandler(deps.dataframe_references)
    text_parts: dict[int, tuple[TextDisplayPart, Any]] = {}
    tool_parts: dict[str, tuple[ToolDisplayPart, Any]] = {}

    async with agent_instance.run_stream_events(
        prompt,
        deps=deps,
        model=model,
        model_settings=model_settings,
        usage_limits=usage_limits,
        message_history=message_history,
    ) as events:
        async for event in events:
            if isinstance(event, PartStartEvent) and isinstance(
                event.part,
                TextPart,
            ):
                part = TextDisplayPart(content=event.part.content)
                assistant_message.parts.append(part)
                placeholder = container.empty()
                placeholder.markdown(part.content)
                text_parts[event.index] = (part, placeholder)

            elif (
                isinstance(event, PartDeltaEvent)
                and isinstance(event.delta, TextPartDelta)
                and event.index in text_parts
            ):
                part, placeholder = text_parts[event.index]
                part.content += event.delta.content_delta
                placeholder.markdown(part.content)

            elif isinstance(event, PartEndEvent):
                text_parts.pop(event.index, None)

            elif isinstance(event, FunctionToolCallEvent):
                call = event.part
                tool_id = call.tool_call_id
                display_name = TOOL_TO_DISPLAY_NAME.get(
                    call.tool_name,
                    call.tool_name.replace("_", " ").title(),
                )
                part = ToolDisplayPart(
                    tool_id=tool_id,
                    title=display_name,
                    args=format_json_string(call.args_as_json_str()),
                )
                assistant_message.parts.append(part)
                status = render_tool(container, part, live=True)
                tool_parts[tool_id] = (part, status)

            elif isinstance(event, FunctionToolResultEvent):
                tool_id = event.tool_call_id
                if tool_id not in tool_parts:
                    continue

                part, status = tool_parts[tool_id]
                if isinstance(event.part, RetryPromptPart):
                    _finish_tool(
                        status,
                        part,
                        output=json_for_display(event.part.content),
                        state="error",
                    )
                    continue

                output = json_for_display(event.part.content)
                try:
                    dataframe_part = await dataframes.render_tool_result(
                        event.part.metadata,
                        container,
                    )
                    if dataframe_part is not None:
                        assistant_message.parts.append(dataframe_part)
                except Exception as exc:
                    _finish_tool(
                        status,
                        part,
                        output=f"Failed to display dataframe: {exc}",
                        state="error",
                    )
                    raise

                _finish_tool(
                    status,
                    part,
                    output=output,
                    state="complete",
                )

            elif isinstance(event, AgentRunResultEvent):
                return StreamResult(
                    assistant_message=assistant_message,
                    model_messages=event.result.all_messages(),
                    artifacts=dataframes.artifacts,
                )

    return None

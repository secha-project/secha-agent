from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pydantic_ai import (
    AgentRunResultEvent,
    ModelMessage,
    ModelSettings,
    UsageLimits,
)

from src.agent.deps import Deps
from src.agent.outputs.artifacts import Artifact

from .models import AssistantDisplayMessage
from .response_events import ResponseEventRenderer


@dataclass(frozen=True)
class StreamResult:
    assistant_message: AssistantDisplayMessage
    model_messages: list[ModelMessage]
    artifacts: tuple[Artifact, ...]


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
    renderer = ResponseEventRenderer(container, deps.artifacts)

    async with agent_instance.run_stream_events(
        prompt,
        deps=deps,
        model=model,
        model_settings=model_settings,
        usage_limits=usage_limits,
        message_history=message_history,
    ) as events:
        async for event in events:
            if isinstance(event, AgentRunResultEvent):
                return StreamResult(
                    assistant_message=renderer.message,
                    model_messages=event.result.all_messages(),
                    artifacts=deps.artifacts.published,
                )
            renderer.render(event)

    return None

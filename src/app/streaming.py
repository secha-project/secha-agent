from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from pydantic_ai import (
    AgentRunResultEvent,
    ModelMessage,
)

from src.agent.outputs.artifacts import Artifact, ArtifactCollector

from .models import AssistantDisplayMessage
from .response_events import ResponseEventRenderer


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

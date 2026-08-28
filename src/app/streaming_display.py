from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel

from .models import ToolDisplayPart


@dataclass(frozen=True)
class ToolStatusView:
    status: Any
    details: Any


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


def render_tool(
    container: Any,
    part: ToolDisplayPart,
    *,
    live: bool = False,
) -> ToolStatusView:
    status = container.status(
        f"Used {part.title}",
        expanded=live or part.state != "complete",
        state=part.state,
        type="compact",
    )
    details = status.container(gap="xsmall")
    details.caption("Parameters")
    details.code(part.args, language="json")
    if part.output is not None:
        heading = "Error" if part.state == "error" else "Output"
        details.caption(heading)
        details.code(part.output, language="text" if part.state == "error" else "json")
    return ToolStatusView(status=status, details=details)


def finish_tool(
    view: ToolStatusView,
    part: ToolDisplayPart,
    *,
    output: str,
    state: Literal["complete", "error"],
) -> None:
    part.output = output
    part.state = state
    heading = "Error" if state == "error" else "Output"
    view.details.caption(heading)
    view.details.code(output, language="text" if state == "error" else "json")
    view.status.update(
        label=f"Used {part.title}",
        state=state,
        expanded=state == "error",
    )

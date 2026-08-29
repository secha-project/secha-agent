from __future__ import annotations

import json
from dataclasses import dataclass
from textwrap import wrap
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel

from src.agent.outputs.artifacts import (
    Artifact,
    ArtifactCollector,
    ArtifactRef,
    ChartArtifact,
    TableArtifact,
)

from .models import ArtifactDisplayPart, ToolDisplayPart


def render_tool_result(
    metadata: object,
    artifacts: ArtifactCollector,
    container: Any,
) -> ArtifactDisplayPart | None:
    if not isinstance(metadata, ArtifactRef):
        return None
    artifact = artifacts.resolve(metadata)
    render_artifact(container, artifact, key=f"download-live-{artifact.id}")
    return ArtifactDisplayPart(artifact_id=artifact.id)


def render_artifact(
    container: Any,
    artifact: Artifact,
    *,
    key: str | None = None,
) -> None:
    container = container.columns([9, 1], gap=None)[0]

    if isinstance(artifact, TableArtifact):
        container.dataframe(artifact.to_pandas(), width="stretch", hide_index=True)
        if artifact.truncated:
            container.caption(
                f"Showing the first {artifact.row_count:,} rows; "
                "the result was truncated."
            )
            container.download_button(
                "Download displayed rows as CSV",
                artifact.to_csv(),
                file_name=f"secha-{artifact.id}.csv",
                mime="text/csv",
                key=key,
            )
        return

    if isinstance(artifact, ChartArtifact):
        spec = artifact.to_spec()
        title = spec.pop("title", None)
        if isinstance(title, str):
            text = wrap(title, width=45) if len(title) > 45 else title
            spec["title"] = {"text": text, "anchor": "middle"}
        elif isinstance(title, dict):
            spec["title"] = {**title, "anchor": "middle"}
        elif title is not None:
            spec["title"] = title
        data_reference = spec.pop("data", None)
        datasets = spec.pop("datasets", None)
        if (
            isinstance(data_reference, dict)
            and isinstance(data_reference.get("name"), str)
            and isinstance(datasets, dict)
            and data_reference["name"] in datasets
        ):
            container.vega_lite_chart(
                pd.DataFrame(datasets[data_reference["name"]]),
                spec,
                width="stretch",
                height=500,
                theme=None,
            )
            return

        if data_reference is not None:
            spec["data"] = data_reference
        if datasets is not None:
            spec["datasets"] = datasets
        container.vega_lite_chart(
            spec,
            width="stretch",
            height=500,
            theme=None,
        )
        return

    raise TypeError(f"No renderer is registered for {type(artifact).__name__}.")


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
        expanded=False,
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
        expanded=False,
    )

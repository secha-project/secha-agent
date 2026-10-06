from __future__ import annotations

import asyncio
import re
from collections.abc import Mapping
from dataclasses import replace
from textwrap import wrap
from typing import Any

import pandas as pd
from pydantic_ai import BinaryContent, ModelRetry, RunContext, ToolReturn
from pydantic_ai.capabilities import Capability
from pydantic_ai.tools import ToolDefinition
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..outputs.artifacts import ChartArtifact
from ..outputs.charts import ChartOutput, ChartOutputRef, UnknownChartOutputError
from ..outputs.tables import TableOutputRef, UnknownTableOutputError
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type

MAX_VISUALIZATION_ROWS = 1_000
MAX_VISUALIZATION_ERROR_LENGTH = 1_500
_DRAW_PATTERN = re.compile(r"\bDRAW\b", re.IGNORECASE)
_TOP_LEVEL_SOURCE_PATTERN = re.compile(
    r"\bVISUALISE\b(?:(?!\b(?:DRAW|SCALE|FACET|PROJECT|LABEL|SETTING|RENAMING)\b).)*\bFROM\b",
    re.IGNORECASE | re.DOTALL,
)
_LAYER_SOURCE_PATTERN = re.compile(
    r"\bMAPPING\b(?:(?!\b(?:DRAW|SCALE|FACET|PROJECT|LABEL|SETTING|RENAMING|FILTER|PARTITION)\b).)*\bFROM\b",
    re.IGNORECASE | re.DOTALL,
)

NAME = "visualization"
# This prompt will only be sent when using deferred loading of capabilities.
DESCRIPTION = "Preview, review, and publish charts from saved table outputs."


def build(
    selected: frozenset[str],
    *,
    prompt_overrides: Mapping[str, str] | None = None,
) -> Capability[Deps]:
    toolset = FunctionToolset[Deps]()
    toolset.add_function(
        visualize_table_output,
        description=render_prompt(
            f"capabilities/{NAME}/tools/visualize_table_output.md",
            selected,
            overrides=prompt_overrides,
        ),
        retries=3,
        sequential=True,
    )
    toolset.add_function(
        display_chart_output,
        description=render_prompt(
            f"capabilities/{NAME}/tools/display_chart_output.md",
            selected,
            overrides=prompt_overrides,
        ),
        prepare=_prepare_display_chart_output,
        sequential=True,
    )
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(
            f"capabilities/{NAME}/instructions.md",
            selected,
            overrides=prompt_overrides,
        ),
        toolsets=[toolset],
    )


def _concise_visualization_error(error: object) -> str:
    message = " ".join(str(error).split())
    if len(message) <= MAX_VISUALIZATION_ERROR_LENGTH:
        return message
    return f"{message[: MAX_VISUALIZATION_ERROR_LENGTH - 1].rstrip()}…"


def normalize_datetime_precision(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize pandas datetime columns to ggsql's microsecond precision."""
    normalized = frame
    for column in frame.columns:
        series = frame[column]
        dtype = series.dtype
        if isinstance(dtype, pd.DatetimeTZDtype):
            if normalized is frame:
                normalized = frame.copy()
            normalized[column] = (
                series.dt.tz_convert("UTC")
                .dt.tz_localize(None)
                .astype("datetime64[us]")
            )
        elif pd.api.types.is_datetime64_dtype(dtype):
            if normalized is frame:
                normalized = frame.copy()
            normalized[column] = series.astype("datetime64[us]")
    return normalized


def _ggsql_errors(validated: Any) -> str:
    messages: list[str] = []
    for error in validated.errors():
        if isinstance(error, dict):
            message = error.get("message")
        else:
            message = getattr(error, "message", None)
        messages.append(str(message or error))
    return "\n".join(messages) or "The ggsql statement is invalid."


def _validate_ggsql(ggsql_text: str) -> str:
    import ggsql as ggsql_package

    if not ggsql_text.strip():
        raise ModelRetry("Visualization validation failed: ggsql text is empty.")
    try:
        validated = ggsql_package.validate(ggsql_text)
    except Exception as exc:
        raise ModelRetry(
            f"Visualization validation failed: {_concise_visualization_error(exc)}"
        ) from exc

    if not validated.valid():
        raise ModelRetry(
            "Visualization validation failed: "
            f"{_concise_visualization_error(_ggsql_errors(validated))}"
        )
    if not validated.has_visual():
        raise ModelRetry("Visualization validation failed: include a VISUALISE clause.")

    visual = validated.visual().strip()
    if not _DRAW_PATTERN.search(visual):
        raise ModelRetry(
            "Visualization validation failed: include at least one DRAW clause."
        )
    if _TOP_LEVEL_SOURCE_PATTERN.search(visual) or _LAYER_SOURCE_PATTERN.search(visual):
        raise ModelRetry(
            "Visualization validation failed: visualization-level FROM sources are "
            "not supported. Build one final relation in the Spark SQL SELECT and "
            "draw every layer from that result."
        )
    return visual


def _use_utc_scales(spec: dict[str, Any]) -> None:
    """Keep ggsql's UTC data, scale bounds, and ticks in the same timezone."""
    for encoding in spec.get("encoding", {}).values():
        if not isinstance(encoding, dict) or encoding.get("type") != "temporal":
            continue
        scale = encoding.setdefault("scale", {})
        if isinstance(scale, dict):
            scale["type"] = "utc"
        for component, key in (
            (scale, "domain"),
            (encoding.get("axis"), "values"),
            (encoding.get("legend"), "values"),
        ):
            if isinstance(component, dict) and isinstance(component.get(key), list):
                # ggsql emits ISO datetime bounds without the Z used in its data.
                component[key] = [
                    value + "Z"
                    if isinstance(value, str)
                    and re.fullmatch(
                        r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?", value
                    )
                    else value
                    for value in component[key]
                ]
    for key in ("layer", "concat", "hconcat", "vconcat"):
        for child in spec.get(key, []):
            _use_utc_scales(child)
    if isinstance(spec.get("spec"), dict):
        _use_utc_scales(spec["spec"])


def _standalone_spec(chart: Any) -> dict[str, Any]:
    spec = chart.to_dict()
    _use_utc_scales(spec)
    if spec.get("width") == "container":
        spec["width"] = 900
    if spec.get("height") == "container":
        spec["height"] = 500
    title = spec.get("title")
    if isinstance(title, str):
        spec["title"] = {
            "text": wrap(title, width=45) if len(title) > 45 else title,
            "anchor": "middle",
        }
    return spec


def _render_preview(spec: dict[str, Any]) -> bytes:
    import vl_convert as vlc

    return vlc.vegalite_to_png(spec, scale=1, allowed_base_urls=[])


async def visualize_table_output(
    ctx: RunContext[Deps],
    ref: TableOutputRef,
    ggsql: str,
    title: str,
) -> ToolReturn[ChartOutput]:
    """Preview a chart from a saved table output and visual-only ggsql.

    Args:
        ref: A reference returned by a tool that produced a table output.
        ggsql: Visual-only syntax beginning with ``VISUALISE`` and containing at
            least one ``DRAW``. Follow the syntax and common-error rules in this
            tool description and reference only columns in the saved output.
            Put the title in ``title``, never ``LABEL title``.
        title: A short, user-friendly chart title.
    """
    if not title.strip():
        raise ModelRetry("Visualization validation failed: title cannot be empty.")

    visual = _validate_ggsql(ggsql)

    deps = ctx.deps
    try:
        output = deps.table_outputs.resolve(ref)
    except UnknownTableOutputError as exc:
        raise ModelRetry(
            f"Table output {ref.output_id!r} is no longer available. "
            "Rerun the tool that produced this table output; do not retry with "
            "this reference."
        ) from exc

    try:
        pandas_frame, truncated = await output.sample(MAX_VISUALIZATION_ROWS)
    except ModelRetry:
        raise
    except Exception as exc:
        raise ModelRetry(
            f"Visualization data loading failed ({spark_error_type(exc)}): "
            f"{concise_spark_error(exc)}."
        ) from exc

    pandas_frame.attrs = {}
    pandas_frame = normalize_datetime_precision(pandas_frame)

    try:
        import ggsql as ggsql_package

        chart = ggsql_package.render_altair(pandas_frame, visual).properties(
            title=title.strip()
        )
        spec = _standalone_spec(chart)
        preview = await asyncio.to_thread(_render_preview, spec)
    except Exception as exc:
        raise ModelRetry(
            f"Visualization rendering failed: {_concise_visualization_error(exc)}"
        ) from exc

    chart_ref = deps.chart_outputs.add(ChartArtifact(spec=spec))

    row_note = (
        f" The chart uses the first {MAX_VISUALIZATION_ROWS:,} rows because the "
        "result exceeded the visualization limit. Aggregate in SQL if this hides "
        "important structure."
        if truncated
        else ""
    )
    return ToolReturn(
        return_value=ChartOutput(output_ref=chart_ref, truncated=truncated),
        content=[
            (
                f"Preview of chart output {chart_ref.output_id}. Not published yet. "
                "Inspect the image for correctness and readability. Revise with "
                "visualize_table_output if needed, then publish the selected preview "
                f"with display_chart_output.{row_note}"
            ),
            BinaryContent(data=preview, media_type="image/png"),
        ],
    )


async def _prepare_display_chart_output(
    _ctx: RunContext[Deps], tool_def: ToolDefinition
) -> ToolDefinition:
    # Keep the reference nested, matching display_table_output's schema.
    return replace(
        tool_def,
        parameters_json_schema={
            "type": "object",
            "properties": {"ref": tool_def.parameters_json_schema},
            "required": ["ref"],
            "additionalProperties": False,
        },
    )


async def display_chart_output(
    ctx: RunContext[Deps], ref: ChartOutputRef
) -> ToolReturn[str]:
    """Publish a chart after reviewing its rendered preview.

    Args:
        ref: A chart-output reference returned by visualize_table_output in this run.
    """
    try:
        artifact = ctx.deps.chart_outputs.resolve(ref)
    except UnknownChartOutputError as exc:
        raise ModelRetry(
            f"Chart output {ref.output_id!r} is no longer available. "
            "Call visualize_table_output to create and review a new preview; "
            "do not retry with this reference."
        ) from exc

    artifact_ref = ctx.deps.artifacts.publish(artifact)
    return ToolReturn(
        return_value=(
            f"Chart artifact {artifact.id} was published. "
            "Continue with any explanation or other requested parts of the answer. "
            "If nothing else was requested, briefly confirm that it is available."
        ),
        metadata=artifact_ref,
    )

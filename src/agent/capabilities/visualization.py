from __future__ import annotations

import re
from typing import Any

import pandas as pd
from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..artifacts.base import ArtifactRef, UnknownArtifactError
from ..artifacts.table import SparkTableSource, TableContent
from ..artifacts.visualization import VisualizationContent
from ..deps import Deps
from ..prompt import render_prompt
from ..spark_runtime import (
    concise_spark_error,
    run_spark_with_timeout,
    spark_error_type,
)

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
DESCRIPTION = "Create charts from saved table results."
visualization_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(
            f"capabilities/{NAME}/instructions.md",
            selected,
        ),
        toolsets=[visualization_toolset],
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


def _standalone_spec(chart: Any) -> dict[str, Any]:
    spec = chart.to_dict()
    if spec.get("height") == "container":
        spec["height"] = 420
    return spec


@visualization_toolset.tool(retries=3, sequential=True)
async def visualize_table_result(
    ctx: RunContext[Deps],
    ref: ArtifactRef,
    ggsql: str,
    title: str,
) -> ToolReturn[str]:
    """Create a chart from a saved table result and visual-only ggsql.

    Prepare filtering, joins, transformations, aggregation, ordering, types, and
    aliases in the producing query. Publish the table only if also requested.

    Use ordered bars for categories, lines for time, histogram or density for a
    distribution, points for two numeric variables, and stacked bars for
    part-to-whole. Avoid pies. Aggregate or bin large data; use opacity when
    overlapping points matter. Use aesthetics only when meaningful and keep
    categorical color to at most seven groups.

    ggsql syntax and common-error rules:

    - Start with ``VISUALISE`` mappings such as ``column AS x`` and add at least
      one ``DRAW mark``. Common marks: ``bar``, ``line``, ``point``,
      ``histogram``, ``density``, and ``tile``.
    - Use ``LABEL aesthetic => 'Label'`` with units. For long categories, use
      ``PROJECT y, x TO cartesian``.
    - For multiple values, use one ``DRAW`` per value with
      ``MAPPING value AS y, 'Series' AS stroke``. Never invent ``y2`` or ``y3``.
    - ``SCALE x VIA date`` accepts dates, never timestamps. Leave timestamp
      scales implicit or use ``datetime`` when needed.
    - Map result columns only: no expressions, functions, or ``RENAMING`` for
      coercion or date formatting. Prepare typed display fields upstream.
    - Do not add ``FROM`` or ``_data``; ``ref`` supplies the data.

    On ModelRetry, correct the syntax or columns with the same result unless its
    data is wrong. Do not generate cosmetic alternatives after success.

    Args:
        ref: A reference returned by a tool that produced a table result.
        ggsql: Visual-only syntax beginning with ``VISUALISE`` and containing at
            least one ``DRAW``. Follow the syntax and common-error rules in this
            tool description and reference only columns in the saved result.
            Put the title in ``title``, never ``LABEL title``.
        title: A short, user-friendly chart title.
    """
    if not title.strip():
        raise ModelRetry("Visualization validation failed: title cannot be empty.")

    visual = _validate_ggsql(ggsql)

    deps = ctx.deps
    try:
        artifact = deps.artifacts.registry.resolve(ref)
    except (UnknownArtifactError, TypeError) as exc:
        raise ModelRetry(
            f"Table artifact {ref.artifact_id!r} is no longer available. "
            "Rerun the tool that produced this table result; do not retry with "
            "this reference."
        ) from exc
    if artifact.kind != "table":
        raise ModelRetry(f"Artifact {ref.artifact_id!r} is not a table artifact.")

    try:
        if isinstance(artifact.source, SparkTableSource):
            pandas_frame = await run_spark_with_timeout(
                artifact.source.spark,
                artifact.source.dataframe.limit(MAX_VISUALIZATION_ROWS + 1).toPandas,
            )
            truncated = len(pandas_frame.index) > MAX_VISUALIZATION_ROWS
        elif isinstance(artifact.content, TableContent):
            pandas_frame = artifact.content.to_pandas()
            truncated = (
                artifact.content.truncated
                or len(pandas_frame.index) > MAX_VISUALIZATION_ROWS
            )
        else:
            raise ModelRetry(
                f"The live source for table artifact {ref.artifact_id!r} is no "
                "longer available. Rerun the tool that produced this table result; "
                "do not retry with this reference."
            )
    except ModelRetry:
        raise
    except Exception as exc:
        raise ModelRetry(
            f"Visualization data loading failed ({spark_error_type(exc)}): "
            f"{concise_spark_error(exc)}."
        ) from exc

    pandas_frame = pandas_frame.iloc[:MAX_VISUALIZATION_ROWS].reset_index(drop=True)
    pandas_frame.attrs = {}
    pandas_frame = normalize_datetime_precision(pandas_frame)

    try:
        import ggsql as ggsql_package

        chart = ggsql_package.render_altair(pandas_frame, visual).properties(
            title=title.strip()
        )
        spec = _standalone_spec(chart)
    except Exception as exc:
        raise ModelRetry(
            f"Visualization rendering failed: {_concise_visualization_error(exc)}"
        ) from exc

    visualization_ref = deps.artifacts.create_content(
        VisualizationContent(spec=spec)
    )
    await deps.artifacts.publish(visualization_ref)

    row_note = (
        f" The chart uses the first {MAX_VISUALIZATION_ROWS:,} rows because the "
        "result exceeded the visualization limit. Aggregate in SQL if this hides "
        "important structure."
        if truncated
        else ""
    )
    return ToolReturn(
        return_value=f"Visualization rendered successfully.{row_note}",
        metadata=visualization_ref,
    )

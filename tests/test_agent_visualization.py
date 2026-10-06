import asyncio
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from pydantic_ai import BinaryContent, ModelRetry

from src.agent.capabilities.visualization import (
    MAX_VISUALIZATION_ROWS,
    display_chart_output,
    normalize_datetime_precision,
    visualize_table_output,
)
from src.agent.outputs.artifacts import ArtifactCollector, ArtifactRef, ChartArtifact
from src.agent.outputs.charts import ChartOutputRef, ChartOutputStore
from src.agent.outputs.tables import (
    LiveTableOutput,
    TableOutputRef,
    TableOutputStore,
)

GGSQL_QUERY = """
VISUALISE category AS x, value AS y
DRAW bar
LABEL x => 'Category', y => 'Value (kW)'
"""


class VisualizationResult:
    schema = object()

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame

    def limit(self, _value: int) -> "VisualizationResult":
        return self

    def toPandas(self) -> pd.DataFrame:
        return self.frame


class VisualizationSpark:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.result = VisualizationResult(frame)
        self.queries: list[str] = []

    def sql(self, sql: str) -> VisualizationResult:
        self.queries.append(sql)
        return self.result

    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


def _context(
    *,
    rows: int = 2,
) -> tuple[Any, VisualizationSpark, TableOutputRef]:
    spark = VisualizationSpark(
        pd.DataFrame(
            {
                "category": [f"category-{index}" for index in range(rows)],
                "value": list(range(rows)),
            }
        )
    )
    table_outputs = TableOutputStore()
    ref = table_outputs.add(
        LiveTableOutput(spark=cast(Any, spark), dataframe=cast(Any, spark.result))
    )
    artifacts = ArtifactCollector()
    return (
        SimpleNamespace(
            deps=SimpleNamespace(
                spark=spark,
                artifacts=artifacts,
                table_outputs=table_outputs,
                chart_outputs=ChartOutputStore(),
            ),
        ),
        spark,
        ref,
    )


def test_visualization_previews_before_publishing_saved_spec() -> None:
    ctx, spark, ref = _context()

    result = asyncio.run(
        visualize_table_output(ctx, ref, GGSQL_QUERY, "Power by category")
    )

    assert result.metadata is None
    assert ctx.deps.artifacts.published == ()
    assert isinstance(result.content[1], BinaryContent)
    assert result.content[1].media_type == "image/png"
    assert result.content[1].data.startswith(b"\x89PNG\r\n\x1a\n")
    artifact = ctx.deps.chart_outputs.resolve(result.return_value.output_ref)
    assert isinstance(artifact, ChartArtifact)
    previewed_spec = artifact.spec_json

    # Publishing uses the exact previewed data even if the source has changed.
    spark.result.frame["value"] = [100, 200]
    published = asyncio.run(display_chart_output(ctx, result.return_value.output_ref))
    assert isinstance(published.metadata, ArtifactRef)
    assert published.content is None
    assert ctx.deps.artifacts.published == (artifact,)
    assert ctx.deps.artifacts.resolve(published.metadata) is artifact
    assert artifact.spec_json == previewed_spec
    assert spark.queries == []


def test_visualization_reports_row_truncation() -> None:
    ctx, _, ref = _context(rows=MAX_VISUALIZATION_ROWS + 1)

    result = asyncio.run(visualize_table_output(ctx, ref, GGSQL_QUERY, "Limited chart"))

    assert result.return_value.truncated is True
    assert "first 1,000 rows" in result.content[0]


def test_only_selected_revision_is_published() -> None:
    ctx, _, ref = _context()
    first = asyncio.run(visualize_table_output(ctx, ref, GGSQL_QUERY, "First draft"))
    revised = asyncio.run(
        visualize_table_output(ctx, ref, GGSQL_QUERY, "Revised chart")
    )

    assert first.return_value.output_ref != revised.return_value.output_ref
    assert ctx.deps.artifacts.published == ()
    asyncio.run(display_chart_output(ctx, revised.return_value.output_ref))
    (artifact,) = ctx.deps.artifacts.published
    assert artifact is ctx.deps.chart_outputs.resolve(revised.return_value.output_ref)


def test_display_rejects_stale_or_table_reference() -> None:
    ctx, _, table_ref = _context()
    for output_id in ("stale", table_ref.output_id):
        with pytest.raises(ModelRetry, match="create and review a new preview"):
            asyncio.run(display_chart_output(ctx, ChartOutputRef(output_id=output_id)))
    assert ctx.deps.artifacts.published == ()


def test_preview_rendering_failure_is_retryable_and_cannot_be_published(monkeypatch):
    ctx, _, ref = _context()
    stored = []
    monkeypatch.setattr(ctx.deps.chart_outputs, "add", stored.append)

    def fail(*_args, **_kwargs):
        raise ValueError("PNG rendering failed")

    monkeypatch.setattr("vl_convert.vegalite_to_png", fail)
    with pytest.raises(ModelRetry, match="Visualization rendering failed.*PNG"):
        asyncio.run(visualize_table_output(ctx, ref, GGSQL_QUERY, "Broken preview"))
    assert stored == []
    assert ctx.deps.artifacts.published == ()


@pytest.mark.parametrize(
    ("ggsql_text", "message"),
    [
        ("", "ggsql text is empty"),
        ("SELECT 1", "include a VISUALISE clause"),
        (
            "VISUALISE category AS x FROM other DRAW bar",
            "visualization-level FROM sources",
        ),
        (
            "VISUALISE category AS x",
            "at least one DRAW",
        ),
        (
            (
                "VISUALISE category AS x, value AS y DRAW bar "
                "MAPPING category AS fill FROM other"
            ),
            "visualization-level FROM sources",
        ),
    ],
)
def test_visualization_validation_failures_are_model_retries(
    ggsql_text: str,
    message: str,
) -> None:
    ctx, _, ref = _context()

    with pytest.raises(ModelRetry, match=message):
        asyncio.run(visualize_table_output(ctx, ref, ggsql_text, "Title"))


def test_datetime_normalization_uses_microseconds_and_utc() -> None:
    frame = pd.DataFrame(
        {
            "naive": pd.to_datetime(["2025-01-01T00:00:00.123456789"]),
            "aware": pd.to_datetime(["2025-01-01T02:00:00.123456789+02:00"]),
            "value": [1],
        }
    )

    normalized = normalize_datetime_precision(frame)

    assert str(normalized["naive"].dtype) == "datetime64[us]"
    assert str(normalized["aware"].dtype) == "datetime64[us]"
    assert normalized.loc[0, "aware"] == pd.Timestamp("2025-01-01T00:00:00.123456")
    assert normalized.loc[0, "value"] == 1


@pytest.mark.parametrize("day", ["2025-03-02", "2025-07-02", "2025-03-30"])
def test_datetime_chart_renders_identically_in_utc_and_helsinki(day: str) -> None:
    ctx, spark, ref = _context()
    spark.result.frame = pd.DataFrame(
        {"time": pd.date_range(day, periods=24, freq="h"), "value": range(24)}
    )
    result = asyncio.run(
        visualize_table_output(
            ctx, ref, "VISUALISE time AS x, value AS y DRAW line", "UTC times"
        )
    )
    spec = ctx.deps.chart_outputs.resolve(result.return_value.output_ref).to_spec()
    spec["width"] = 600
    render = (
        "import json, sys, vl_convert; "
        "print(vl_convert.vegalite_to_svg(json.load(sys.stdin)))"
    )
    svgs = [
        subprocess.run(
            [sys.executable, "-c", render],
            input=json.dumps(spec),
            text=True,
            capture_output=True,
            check=True,
            env={**os.environ, "TZ": timezone},
        ).stdout
        for timezone in ("UTC", "Europe/Helsinki")
    ]
    assert svgs[0] == svgs[1]


def test_visualization_rejects_stale_table_reference() -> None:
    ctx, _, _ = _context()

    with pytest.raises(ModelRetry, match="Rerun the tool"):
        asyncio.run(
            visualize_table_output(
                ctx,
                TableOutputRef(output_id="stale"),
                GGSQL_QUERY,
                "Stale",
            )
        )


def test_visualization_exposes_spark_failure_as_model_retry() -> None:
    class BrokenResult(VisualizationResult):
        def toPandas(self) -> pd.DataFrame:
            raise RuntimeError("backend unavailable")

    spark = VisualizationSpark(pd.DataFrame())
    spark.result = BrokenResult(pd.DataFrame())
    table_outputs = TableOutputStore()
    ref = table_outputs.add(
        LiveTableOutput(spark=cast(Any, spark), dataframe=cast(Any, spark.result))
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=spark,
            artifacts=ArtifactCollector(),
            table_outputs=table_outputs,
        )
    )

    with pytest.raises(
        ModelRetry,
        match="Visualization data loading failed.*backend unavailable",
    ):
        asyncio.run(visualize_table_output(cast(Any, ctx), ref, GGSQL_QUERY, "Broken"))

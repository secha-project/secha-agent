import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from pydantic_ai import ModelRetry

from src.agent.artifacts.base import (
    ArtifactRef,
    RunArtifacts,
)
from src.agent.artifacts.table import SparkTableSource
from src.agent.artifacts.visualization import VisualizationContent
from src.agent.capabilities.visualization import (
    MAX_VISUALIZATION_ROWS,
    normalize_datetime_precision,
    visualization_toolset,
    visualize_table_result,
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
        self.limit_value: int | None = None

    def limit(self, value: int) -> "VisualizationResult":
        self.limit_value = value
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
) -> tuple[Any, VisualizationSpark, ArtifactRef]:
    spark = VisualizationSpark(
        pd.DataFrame(
            {
                "category": [f"category-{index}" for index in range(rows)],
                "value": list(range(rows)),
            }
        )
    )
    artifacts = RunArtifacts()
    ref = artifacts.create(
        SparkTableSource(spark=cast(Any, spark), dataframe=cast(Any, spark.result))
    )
    return (
        SimpleNamespace(
            deps=SimpleNamespace(spark=spark, artifacts=artifacts),
        ),
        spark,
        ref,
    )


def test_visualization_tool_exposes_retry_contract_in_definition() -> None:
    tool = visualization_toolset.tools["visualize_table_result"]
    description = " ".join((tool.description or "").split())

    assert tool.max_retries == 3
    assert tool.sequential is True
    assert "ModelRetry" in description
    assert "ordered bars for categories" in description
    assert "Aggregate or bin large data" in description
    assert "PROJECT y, x TO cartesian" in description
    assert "Never invent ``y2`` or ``y3``" in description
    assert "``SCALE x VIA date`` accepts dates" in description
    assert "never timestamps" in description
    assert "no expressions, functions, or ``RENAMING``" in description
    assert "Do not add ``FROM`` or ``_data``" in description
    ggsql_description = " ".join(
        tool.function_schema.json_schema["properties"]["ggsql"]["description"].split()
    )
    assert "beginning with ``VISUALISE``" in ggsql_description
    assert "at least one ``DRAW``" in ggsql_description
    assert tool.function_schema.json_schema["required"] == ["ref", "ggsql", "title"]


def test_visualization_publishes_registered_spec() -> None:
    ctx, spark, ref = _context()

    result = asyncio.run(
        visualize_table_result(ctx, ref, GGSQL_QUERY, "Power by category")
    )

    assert isinstance(result.metadata, ArtifactRef)
    assert result.content is None
    artifact = ctx.deps.artifacts.published[0]
    assert ctx.deps.artifacts.registry.resolve(result.metadata) is artifact
    content = artifact.require_content()
    assert isinstance(content, VisualizationContent)
    assert spark.result.limit_value == MAX_VISUALIZATION_ROWS + 1
    assert spark.queries == []


def test_visualization_reports_row_truncation() -> None:
    ctx, _, ref = _context(rows=MAX_VISUALIZATION_ROWS + 1)

    result = asyncio.run(visualize_table_result(ctx, ref, GGSQL_QUERY, "Limited chart"))

    assert "first 1,000 rows" in result.return_value


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
        asyncio.run(visualize_table_result(ctx, ref, ggsql_text, "Title"))


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


def test_visualization_rejects_stale_table_reference() -> None:
    ctx, _, _ = _context()

    with pytest.raises(ModelRetry, match="Rerun the tool"):
        asyncio.run(
            visualize_table_result(
                ctx,
                ArtifactRef(artifact_id="stale"),
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
    artifacts = RunArtifacts()
    ref = artifacts.create(
        SparkTableSource(spark=cast(Any, spark), dataframe=cast(Any, spark.result))
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=spark,
            artifacts=artifacts,
        )
    )

    with pytest.raises(
        ModelRetry,
        match="Visualization data loading failed.*backend unavailable",
    ):
        asyncio.run(visualize_table_result(cast(Any, ctx), ref, GGSQL_QUERY, "Broken"))

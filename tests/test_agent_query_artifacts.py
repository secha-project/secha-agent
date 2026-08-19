import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from pydantic_ai import ModelRetry

from src.agent.artifacts.base import (
    ArtifactPolicy,
    ArtifactRef,
    RunArtifacts,
)
from src.agent.artifacts.table import SparkTableSource, TableContent
from src.agent.capabilities.query import (
    InspectQueryResult,
    display_query_result,
    inspect_query_result,
    query_data,
)


class StubSchema:
    def fieldNames(self) -> list[str]:
        return ["known_column"]


class StubResult:
    schema = StubSchema()

    def limit(self, _value: int) -> "StubResult":
        return self

    def count(self) -> int:
        return 1

    def select(self, *_columns: str) -> "StubResult":
        return self

    def collect(self) -> list[tuple[int]]:
        return [(1,)]

    def toPandas(self) -> pd.DataFrame:
        return pd.DataFrame([{"known_column": 1}])


class StubSpark:
    def sql(self, _sql: str) -> StubResult:
        return StubResult()

    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


def test_result_preview_uses_framework_agnostic_validation_error() -> None:
    source = SparkTableSource(
        cast(Any, StubSpark()),
        cast(Any, StubResult()),
    )
    with pytest.raises(ValueError, match="Unknown result columns: missing_column"):
        asyncio.run(source.preview(column_names=["missing_column"]))


@pytest.mark.parametrize(
    ("start_row", "end_row"),
    [(0, None), (3, 2)],
)
def test_result_preview_rejects_invalid_row_ranges(
    start_row: int,
    end_row: int | None,
) -> None:
    source = SparkTableSource(cast(Any, StubSpark()), cast(Any, StubResult()))

    with pytest.raises(ValueError):
        asyncio.run(source.preview(start_row=start_row, end_row=end_row))


def test_inspect_query_result_exposes_unknown_columns_as_model_retry() -> None:
    artifacts = RunArtifacts()
    ref = artifacts.create(
        SparkTableSource(cast(Any, StubSpark()), cast(Any, StubResult()))
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            artifacts=artifacts,
        )
    )

    with pytest.raises(ModelRetry, match="Unknown result columns: missing_column"):
        asyncio.run(
            inspect_query_result(
                cast(Any, ctx),
                ref,
                column_names=["missing_column"],
            )
        )


@pytest.mark.parametrize(
    "tool",
    [inspect_query_result, display_query_result],
)
def test_query_result_tools_expose_stale_references_as_model_retry(
    tool: Callable[..., Any],
) -> None:
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            artifacts=RunArtifacts(),
        )
    )

    with pytest.raises(
        ModelRetry,
        match=r"(?:source for table artifact|Table artifact) 'stale'.*Call query_data again",
    ):
        asyncio.run(
            tool(
                cast(Any, ctx),
                ArtifactRef(artifact_id="stale"),
            )
        )


def test_query_data_returns_structured_result_preview() -> None:
    artifacts = RunArtifacts(
        policy=ArtifactPolicy(table_max_rows=10),
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            artifacts=artifacts,
        )
    )

    result = asyncio.run(
        query_data(
            cast(Any, ctx),
            "SELECT 1 AS value",
        )
    )

    assert result.preview_markdown == "## Row 1\n\nknown_column: 1"
    source = artifacts.resolve_source(
        result.artifact_ref,
        SparkTableSource,
    )
    assert isinstance(source.dataframe, StubResult)


def test_display_query_result_publishes_the_same_logical_artifact() -> None:
    artifacts = RunArtifacts(
        policy=ArtifactPolicy(table_max_rows=10),
    )
    ref = artifacts.create(
        SparkTableSource(cast(Any, StubSpark()), cast(Any, StubResult()))
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            artifacts=artifacts,
        )
    )

    result = asyncio.run(
        display_query_result(
            cast(Any, ctx),
            ref,
        )
    )

    assert isinstance(result.metadata, ArtifactRef)
    assert result.metadata == ref
    artifact = artifacts.registry.resolve(result.metadata)
    assert isinstance(artifact.content, TableContent)
    assert artifact.content.row_count == 1


def test_display_query_result_can_publish_materialized_content_without_a_source(
    make_table_artifact,
) -> None:
    artifacts = RunArtifacts()
    artifact = make_table_artifact([{"known_column": 1}])
    artifacts.registry.add(artifact)
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            artifacts=artifacts,
        )
    )

    result = asyncio.run(
        display_query_result(
            cast(Any, ctx),
            artifact.ref,
        )
    )

    assert result.metadata == artifact.ref
    assert artifacts.published == (artifact,)


def test_query_data_rejects_mutating_sql_before_calling_spark() -> None:
    class SparkThatMustNotRun:
        def sql(self, _sql: str) -> StubResult:
            raise AssertionError("Spark should not receive mutating SQL")

    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=SparkThatMustNotRun(),
            artifacts=RunArtifacts(
                policy=ArtifactPolicy(table_max_rows=10),
            ),
        )
    )

    with pytest.raises(ModelRetry, match="ReadOnlyQueryRequired"):
        asyncio.run(
            query_data(
                cast(Any, ctx),
                "DELETE FROM measurements",
            )
        )


def test_inspect_query_result_returns_structured_result_preview() -> None:
    artifacts = RunArtifacts()
    ref = artifacts.create(
        SparkTableSource(cast(Any, StubSpark()), cast(Any, StubResult()))
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            artifacts=artifacts,
        )
    )

    result = asyncio.run(
        inspect_query_result(
            cast(Any, ctx),
            ref,
        )
    )

    assert result == InspectQueryResult(
        artifact_ref=ref,
        preview_markdown="## Row 1\n\nknown_column: 1",
    )

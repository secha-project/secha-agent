import asyncio
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from pydantic_ai import ModelRetry

from src.agent.capabilities.query import query_data
from src.agent.capabilities.table_outputs import (
    display_table_output,
    inspect_table_output,
)
from src.agent.outputs.artifacts import ArtifactCollector, ArtifactRef, TableArtifact
from src.agent.outputs.tables import (
    LiveTableOutput,
    TableOutput,
    TableOutputRef,
    TableOutputStore,
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


def make_deps() -> SimpleNamespace:
    return SimpleNamespace(
        spark=StubSpark(),
        table_outputs=TableOutputStore(),
        artifacts=ArtifactCollector(),
        table_artifact_max_rows=10,
        semantic_layer_enabled=False,
    )


def add_output(deps: SimpleNamespace) -> TableOutputRef:
    return deps.table_outputs.add(
        LiveTableOutput(cast(Any, StubSpark()), cast(Any, StubResult()))
    )


@pytest.mark.parametrize(("start_row", "end_row"), [(0, None), (3, 2)])
def test_output_preview_rejects_invalid_row_ranges(
    start_row: int,
    end_row: int | None,
) -> None:
    output = LiveTableOutput(cast(Any, StubSpark()), cast(Any, StubResult()))
    with pytest.raises(ValueError):
        asyncio.run(output.preview(start_row=start_row, end_row=end_row))


def test_inspect_table_output_exposes_unknown_columns_as_model_retry() -> None:
    deps = make_deps()
    ref = add_output(deps)
    with pytest.raises(ModelRetry, match="Unknown result columns: missing_column"):
        asyncio.run(
            inspect_table_output(
                cast(Any, SimpleNamespace(deps=deps)),
                ref,
                column_names=["missing_column"],
            )
        )


@pytest.mark.parametrize("tool", [inspect_table_output, display_table_output])
def test_table_output_tools_expose_stale_references_as_model_retry(
    tool: Callable[..., Any],
) -> None:
    with pytest.raises(ModelRetry, match=r"Table output 'stale'.*Rerun the tool"):
        asyncio.run(
            tool(
                cast(Any, SimpleNamespace(deps=make_deps())),
                TableOutputRef(output_id="stale"),
            )
        )


def test_query_data_returns_structured_output_preview() -> None:
    deps = make_deps()
    result = asyncio.run(
        query_data(cast(Any, SimpleNamespace(deps=deps)), "SELECT 1 AS value")
    )

    assert result.preview_markdown == "## Row 1\n\nknown_column: 1"
    output = deps.table_outputs.resolve(result.output_ref)
    assert isinstance(output.dataframe, StubResult)


def test_display_table_output_creates_a_concrete_artifact() -> None:
    deps = make_deps()
    result = asyncio.run(
        display_table_output(
            cast(Any, SimpleNamespace(deps=deps)),
            add_output(deps),
        )
    )

    assert isinstance(result.metadata, ArtifactRef)
    artifact = deps.artifacts.resolve(result.metadata)
    assert isinstance(artifact, TableArtifact)
    assert artifact.row_count == 1


def test_query_data_rejects_mutating_sql_before_calling_spark() -> None:
    class SparkThatMustNotRun:
        def sql(self, _sql: str) -> StubResult:
            raise AssertionError("Spark should not receive mutating SQL")

    deps = make_deps()
    deps.spark = SparkThatMustNotRun()
    with pytest.raises(ModelRetry, match="ReadOnlyQueryRequired"):
        asyncio.run(
            query_data(
                cast(Any, SimpleNamespace(deps=deps)),
                "DELETE FROM measurements",
            )
        )


def test_inspect_table_output_returns_structured_output_preview() -> None:
    deps = make_deps()
    ref = add_output(deps)
    result = asyncio.run(
        inspect_table_output(cast(Any, SimpleNamespace(deps=deps)), ref)
    )

    assert result == TableOutput(
        output_ref=ref,
        preview_markdown="## Row 1\n\nknown_column: 1",
    )

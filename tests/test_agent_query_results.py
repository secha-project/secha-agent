import asyncio
from collections.abc import Callable
from types import SimpleNamespace

import pytest
from pydantic_ai import ModelRetry

import src.agent.tools.query_tools as query_tools_module
from src.agent.models import InspectQueryResult, QueryDataResult
from src.agent.query_results import (
    UnknownQueryResultReferenceError,
    render_result_preview,
)
from src.agent.tools.query_tools import (
    display_query_result,
    inspect_query_result,
    query_data,
)


class StubSchema:
    def fieldNames(self) -> list[str]:
        return ["known_column"]


class StubResult:
    schema = StubSchema()


class StubResultRegistry:
    def resolve(self, _ref: str) -> StubResult:
        return StubResult()


class StubQueryResultRegistry(StubResultRegistry):
    def create(self, _result: StubResult) -> str:
        return "df-1"


class StaleResultRegistry:
    def resolve(self, ref: str) -> StubResult:
        raise UnknownQueryResultReferenceError(ref)


class StubSpark:
    def sql(self, _sql: str) -> StubResult:
        return StubResult()


async def run_immediately(_spark: object, operation: Callable[[], str]) -> str:
    return operation()


def test_result_preview_uses_framework_agnostic_validation_error() -> None:
    with pytest.raises(ValueError, match="Unknown result columns: missing_column"):
        render_result_preview(
            StubResult(),  # type: ignore[arg-type]
            column_names=["missing_column"],
        )


def test_inspect_query_result_exposes_unknown_columns_as_model_retry() -> None:
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            result_registry=StubResultRegistry(),
        )
    )

    with pytest.raises(ModelRetry, match="Unknown result columns: missing_column"):
        asyncio.run(
            inspect_query_result(
                ctx,  # type: ignore[arg-type]
                "df-1",
                column_names=["missing_column"],
            )
        )


@pytest.mark.parametrize(
    "tool",
    [inspect_query_result, display_query_result],
)
def test_query_result_tools_expose_stale_references_as_model_retry(
    tool: Callable[..., object],
) -> None:
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            result_registry=StaleResultRegistry(),
            result_exporter=None,
        )
    )

    with pytest.raises(
        ModelRetry,
        match=r"reference 'df-stale' is stale.*Call query_data again",
    ):
        asyncio.run(
            tool(
                ctx,  # type: ignore[arg-type]
                "df-stale",
            )
        )


def test_query_data_returns_structured_result_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_tools_module,
        "run_spark_with_timeout",
        run_immediately,
    )
    monkeypatch.setattr(
        query_tools_module,
        "render_result_preview",
        lambda _result: "| value |\n| --- |\n| 1 |",
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            result_registry=StubQueryResultRegistry(),
        )
    )

    result = asyncio.run(
        query_data(
            ctx,  # type: ignore[arg-type]
            "SELECT 1 AS value",
        )
    )

    assert result == QueryDataResult(
        dataframe_ref="df-1",
        preview_markdown="| value |\n| --- |\n| 1 |",
    )


def test_query_data_rejects_mutating_sql_before_calling_spark() -> None:
    class SparkThatMustNotRun:
        def sql(self, _sql: str) -> StubResult:
            raise AssertionError("Spark should not receive mutating SQL")

    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=SparkThatMustNotRun(),
            result_registry=StubQueryResultRegistry(),
        )
    )

    with pytest.raises(ModelRetry, match="ReadOnlyQueryRequired"):
        asyncio.run(
            query_data(
                ctx,  # type: ignore[arg-type]
                "DELETE FROM measurements",
            )
        )


def test_inspect_query_result_returns_structured_result_preview(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        query_tools_module,
        "run_spark_with_timeout",
        run_immediately,
    )
    monkeypatch.setattr(
        query_tools_module,
        "render_result_preview",
        lambda _result, **_kwargs: "| value |\n| --- |\n| 1 |",
    )
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            result_registry=StubResultRegistry(),
        )
    )

    result = asyncio.run(
        inspect_query_result(
            ctx,  # type: ignore[arg-type]
            "df-1",
        )
    )

    assert result == InspectQueryResult(
        dataframe_ref="df-1",
        preview_markdown="| value |\n| --- |\n| 1 |",
    )

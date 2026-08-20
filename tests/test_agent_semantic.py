import asyncio
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
from pydantic import ValidationError
from pydantic_ai import ModelRetry
from pydantic_ai.models.test import TestModel

import src.agent.semantic_runtime as semantic_module
from src.agent.agent import create_agent
from src.agent.artifacts.base import RunArtifacts
from src.agent.artifacts.table import SparkTableSource
from src.agent.capabilities.semantic_layer import (
    SemanticOrder,
    SemanticTimeRange,
    SetSemanticFilter,
    ValueSemanticFilter,
    query_semantic_model,
    semantic_toolset,
)
from src.agent.semantic_runtime import SemanticRuntime, load_semantic_runtime


class StubSchema:
    def fieldNames(self) -> list[str]:
        return ["total_energy"]


class StubResult:
    schema = StubSchema()

    def limit(self, _value: int) -> "StubResult":
        return self

    def count(self) -> int:
        return 1

    def select(self, *_columns: str) -> "StubResult":
        return self

    def collect(self) -> list[tuple[int]]:
        return [(42,)]

    def toPandas(self) -> pd.DataFrame:
        return pd.DataFrame([{"total_energy": 42}])


class StubSpark:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def sql(self, sql: str) -> StubResult:
        self.queries.append(sql)
        return StubResult()

    def addTag(self, _tag: str) -> None:
        pass

    def removeTag(self, _tag: str) -> None:
        pass


class StubSemanticModel:
    description = "Governed energy metrics"

    def __init__(self) -> None:
        self.query_arguments: dict[str, Any] | None = None

    def query(self, **kwargs: Any) -> object:
        self.query_arguments = kwargs
        return object()


class StubSemanticRuntime:
    def __init__(self, model: StubSemanticModel) -> None:
        self.models = {"energy": model}
        self.expression: object | None = None

    def get_model(self, model_name: str) -> StubSemanticModel:
        if model_name != "energy":
            raise ValueError("unknown semantic model")
        return self.models[model_name]

    def compile(self, expression: object) -> str:
        self.expression = expression
        return "SELECT 42 AS total_energy"


def test_semantic_capability_exposes_structured_tools() -> None:
    model = TestModel(call_tools=[], custom_output_text="done")

    create_agent(("semantic_layer", "table_results")).run_sync(
        "question",
        deps=cast(Any, object()),
        model=model,
    )

    request = model.last_model_request_parameters
    assert request is not None
    assert [tool.name for tool in request.function_tools] == [
        "list_semantic_models",
        "describe_semantic_model",
        "query_semantic_model",
        "compare_semantic_periods",
        "inspect_table_result",
        "display_table_result",
    ]
    query_schema = semantic_toolset.tools["query_semantic_model"].function_schema
    assert "$defs" in query_schema.json_schema
    assert "filters" in query_schema.json_schema["properties"]


def test_semantic_capability_requires_table_results_capability() -> None:
    with pytest.raises(
        ValueError, match="semantic_layer capability requires table_results"
    ):
        create_agent(("semantic_layer",))


def test_semantic_filter_validates_operator_specific_shape() -> None:
    assert SetSemanticFilter(
        field="status",
        operator="in",
        values=["active", "pending"],
    ).as_bsl_filter() == {
        "field": "status",
        "operator": "in",
        "values": ["active", "pending"],
    }

    with pytest.raises(ValidationError, match="Field required"):
        SetSemanticFilter.model_validate({"field": "status", "operator": "in"})

    with pytest.raises(ValidationError, match="at least 1 item"):
        SetSemanticFilter(field="status", operator="in", values=[])


def test_semantic_time_range_requires_ordered_iso_bounds() -> None:
    with pytest.raises(ValidationError, match="start must be before end"):
        SemanticTimeRange(start="2026-02-01", end="2026-01-01")

    with pytest.raises(ValidationError, match="ISO 8601"):
        SemanticTimeRange(start="later", end="earlier")


def test_semantic_query_uses_the_standard_spark_table_artifact_path() -> None:
    spark = StubSpark()
    model = StubSemanticModel()
    runtime = StubSemanticRuntime(model)
    artifacts = RunArtifacts()
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=spark,
            semantic=runtime,
            artifacts=artifacts,
        )
    )

    result = asyncio.run(
        query_semantic_model(
            cast(Any, ctx),
            model_name="energy",
            dimensions=["site"],
            measures=["total_energy"],
            filters=[ValueSemanticFilter(field="site", operator="=", value="A")],
            order_by=[SemanticOrder(field="total_energy", direction="desc")],
            time_range=SemanticTimeRange(
                start="2026-01-01",
                end="2026-02-01",
            ),
        )
    )

    assert spark.queries == ["SELECT 42 AS total_energy"]
    assert result.preview_markdown == "## Row 1\n\ntotal_energy: 42"
    assert model.query_arguments == {
        "dimensions": ["site"],
        "measures": ["total_energy"],
        "filters": [{"field": "site", "operator": "=", "value": "A"}],
        "order_by": [("total_energy", "desc")],
        "limit": None,
        "time_grain": None,
        "time_grains": None,
        "time_range": {"start": "2026-01-01", "end": "2026-02-01"},
    }
    source = artifacts.resolve_source(
        result.artifact_ref,
        SparkTableSource,
    )
    assert isinstance(source.dataframe, StubResult)


def test_semantic_query_reports_unknown_models_as_retry() -> None:
    ctx = SimpleNamespace(
        deps=SimpleNamespace(
            spark=StubSpark(),
            semantic=StubSemanticRuntime(StubSemanticModel()),
            artifacts=RunArtifacts(),
        )
    )

    with pytest.raises(ModelRetry, match="unknown semantic model"):
        asyncio.run(
            query_semantic_model(
                cast(Any, ctx),
                model_name="missing",
                measures=["total_energy"],
            )
        )


def test_runtime_builds_models_with_the_existing_ibis_backend(
    monkeypatch: Any,
) -> None:
    backend = object()
    monkeypatch.setattr(
        semantic_module.ibis.pyspark,
        "connect",
        lambda spark: backend,
    )

    def build_semantic_models(value: object) -> dict[str, Any]:
        assert value is backend
        return {"energy": "semantic-model"}

    monkeypatch.setattr(
        semantic_module,
        "build_semantic_models",
        build_semantic_models,
    )

    runtime = load_semantic_runtime(cast(Any, object()))

    assert isinstance(runtime, SemanticRuntime)
    assert runtime.backend is backend
    assert runtime.models == {"energy": "semantic-model"}


def test_runtime_untags_bsl_expressions_before_compiling(
    monkeypatch: Any,
) -> None:
    expression = object()
    untagged = object()

    class Backend:
        def compile(self, value: object) -> str:
            assert value is untagged
            return "SELECT 1"

    monkeypatch.setattr(
        semantic_module,
        "to_untagged",
        lambda value: untagged if value is expression else None,
    )
    runtime = SemanticRuntime(
        backend=cast(Any, Backend()),
        models={},
    )

    assert runtime.compile(expression) == "SELECT 1"

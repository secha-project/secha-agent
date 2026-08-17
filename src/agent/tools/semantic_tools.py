from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, model_validator
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..spark import concise_spark_error, spark_error_type
from .query_tools import QueryDataResult, create_query_result

semantic_toolset = FunctionToolset[Deps]()


class SemanticModelSummary(BaseModel):
    name: str
    description: str | None


class ListSemanticModelsResult(BaseModel):
    models: list[SemanticModelSummary]


class SemanticDimension(BaseModel):
    name: str
    description: str | None
    is_time_dimension: bool
    smallest_time_grain: str | None


class SemanticMeasure(BaseModel):
    name: str
    description: str | None
    calculated: bool = False


class DescribeSemanticModelResult(BaseModel):
    name: str
    description: str | None
    dimensions: list[SemanticDimension]
    measures: list[SemanticMeasure]


FilterOperator = Literal[
    "=",
    "eq",
    "equals",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "in",
    "not in",
    "like",
    "not like",
    "ilike",
    "not ilike",
    "is null",
    "is not null",
    "AND",
    "OR",
]


class SemanticFilter(BaseModel):
    field: str | None = None
    operator: FilterOperator
    value: Any | None = None
    values: list[Any] | None = None
    conditions: list[SemanticFilter] | None = None

    @model_validator(mode="after")
    def validate_shape(self) -> SemanticFilter:
        if self.operator in {"AND", "OR"}:
            if not self.conditions:
                raise ValueError(
                    f"{self.operator} filters require non-empty conditions."
                )
            if (
                self.field is not None
                or self.value is not None
                or self.values is not None
            ):
                raise ValueError(
                    f"{self.operator} filters only accept the conditions field."
                )
            return self

        if not self.field:
            raise ValueError(f"{self.operator} filters require a field.")
        if self.conditions is not None:
            raise ValueError(
                f"{self.operator} filters do not accept nested conditions."
            )
        if self.operator in {"in", "not in"}:
            if self.values is None:
                raise ValueError(f"{self.operator} filters require values.")
            if self.value is not None:
                raise ValueError(f"{self.operator} filters do not accept value.")
        elif self.operator in {"is null", "is not null"}:
            if self.value is not None or self.values is not None:
                raise ValueError(f"{self.operator} filters do not accept values.")
        elif self.value is None:
            raise ValueError(f"{self.operator} filters require value.")
        elif self.values is not None:
            raise ValueError(f"{self.operator} filters do not accept values.")
        return self

    def as_bsl_filter(self) -> dict[str, Any]:
        return self.model_dump(exclude_none=True)


class SemanticOrder(BaseModel):
    field: str
    direction: Literal["asc", "desc"] = "asc"


class SemanticTimeRange(BaseModel):
    start: str
    end: str


def _runtime(ctx: RunContext[Deps]):
    runtime = ctx.deps.semantic
    if runtime is None:
        raise ModelRetry(
            "The semantic runtime is not configured. Do not retry this tool; "
            "use the catalog and query capabilities instead."
        )
    return runtime


def _model(ctx: RunContext[Deps], model_name: str):
    try:
        return _runtime(ctx).get_model(model_name)
    except ValueError as exc:
        raise ModelRetry(str(exc)) from exc


async def _execute_semantic_expression(
    ctx: RunContext[Deps],
    expression: Any,
) -> QueryDataResult:
    runtime = _runtime(ctx)
    try:
        sql = runtime.compile(expression)
        dataframe = ctx.deps.spark.sql(sql)
        _ = dataframe.schema
    except Exception as exc:
        raise ModelRetry(
            "Semantic query compilation or validation failed "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc

    return await create_query_result(
        ctx.deps,
        dataframe,
        error_prefix="Semantic query execution failed while reading the preview",
    )


@semantic_toolset.tool
def list_semantic_models(ctx: RunContext[Deps]) -> ListSemanticModelsResult:
    """List the governed semantic models and their business descriptions."""
    runtime = _runtime(ctx)
    return ListSemanticModelsResult(
        models=[
            SemanticModelSummary(name=name, description=model.description)
            for name, model in sorted(runtime.models.items())
        ]
    )


@semantic_toolset.tool
def describe_semantic_model(
    ctx: RunContext[Deps],
    model_name: str,
) -> DescribeSemanticModelResult:
    """Describe one semantic model's governed dimensions and measures.

    Args:
        model_name: Exact model name returned by ``list_semantic_models``.
    """
    model = _model(ctx, model_name)
    dimensions = model.get_dimensions()
    measures = model.get_measures()
    calculated_measures = model.get_calculated_measures()
    return DescribeSemanticModelResult(
        name=model_name,
        description=model.description,
        dimensions=[
            SemanticDimension(
                name=name,
                description=dimension.description,
                is_time_dimension=dimension.is_time_dimension,
                smallest_time_grain=dimension.smallest_time_grain,
            )
            for name, dimension in dimensions.items()
        ],
        measures=[
            SemanticMeasure(
                name=name,
                description=measure.description,
            )
            for name, measure in measures.items()
        ]
        + [
            SemanticMeasure(
                name=name,
                description=getattr(measure, "description", None),
                calculated=True,
            )
            for name, measure in calculated_measures.items()
        ],
    )


@semantic_toolset.tool(retries=3, sequential=True)
async def query_semantic_model(
    ctx: RunContext[Deps],
    model_name: str,
    dimensions: list[str] | None = None,
    measures: list[str] | None = None,
    filters: list[SemanticFilter] | None = None,
    order_by: list[SemanticOrder] | None = None,
    limit: Annotated[int | None, Field(ge=1, le=100_000)] = None,
    time_grain: str | None = None,
    time_grains: dict[str, str] | None = None,
    time_range: SemanticTimeRange | None = None,
) -> QueryDataResult:
    """Query governed dimensions and measures from one semantic model.

    The result is saved as the same live table artifact used by ``query_data``.
    Use this instead of recreating a declared business measure in raw SQL.

    Args:
        model_name: Exact model name returned by ``list_semantic_models``.
        dimensions: Governed dimensions used to group the result.
        measures: Governed measures to calculate.
        filters: Dimension or measure filters using BSL filter operators.
        order_by: Selected result fields and their sort directions.
        limit: Maximum result rows.
        time_grain: One grain applied to selected time dimensions.
        time_grains: Per-dimension grains; cannot be combined with time_grain.
        time_range: Inclusive start and exclusive end time range.
    """
    if time_grain is not None and time_grains is not None:
        raise ModelRetry("Use either time_grain or time_grains, not both.")
    model = _model(ctx, model_name)
    try:
        expression = model.query(
            dimensions=dimensions,
            measures=measures,
            filters=[item.as_bsl_filter() for item in filters or []],
            order_by=[(item.field, item.direction) for item in order_by or []],
            limit=limit,
            time_grain=time_grain,
            time_grains=time_grains,
            time_range=time_range.model_dump() if time_range else None,
        )
    except Exception as exc:
        raise ModelRetry(f"Semantic query is invalid: {exc}") from exc
    return await _execute_semantic_expression(ctx, expression)


@semantic_toolset.tool(retries=3, sequential=True)
async def compare_semantic_periods(
    ctx: RunContext[Deps],
    model_name: str,
    measures: list[str],
    current_time_range: SemanticTimeRange,
    previous_time_range: SemanticTimeRange,
    dimensions: list[str] | None = None,
    filters: list[SemanticFilter] | None = None,
    time_dimension: str | None = None,
    time_grain: str | None = None,
    time_grains: dict[str, str] | None = None,
    order_by: list[SemanticOrder] | None = None,
    limit: Annotated[int | None, Field(ge=1, le=100_000)] = None,
) -> QueryDataResult:
    """Compare governed measures across two explicit time ranges.

    The result includes current, previous, delta, and percentage-change columns
    and is saved as the same live table artifact used by ``query_data``.
    """
    if time_grain is not None and time_grains is not None:
        raise ModelRetry("Use either time_grain or time_grains, not both.")
    model = _model(ctx, model_name)
    try:
        expression = model.compare_periods(
            dimensions=dimensions,
            measures=measures,
            current_time_range=current_time_range.model_dump(),
            previous_time_range=previous_time_range.model_dump(),
            filters=[item.as_bsl_filter() for item in filters or []],
            time_dimension=time_dimension,
            time_grain=time_grain,
            time_grains=time_grains,
            order_by=[(item.field, item.direction) for item in order_by or []],
            limit=limit,
        )
    except Exception as exc:
        raise ModelRetry(f"Semantic period comparison is invalid: {exc}") from exc
    return await _execute_semantic_expression(ctx, expression)

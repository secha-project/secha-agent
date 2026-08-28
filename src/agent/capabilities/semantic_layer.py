from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    model_validator,
)
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..outputs.tables import TableOutput, create_table_output
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type

NAME = "semantic_layer"
DESCRIPTION = "Query governed semantic models, measures, and dimensions."
semantic_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[semantic_toolset],
    )


SemanticName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
SemanticNames = Annotated[list[SemanticName], Field(min_length=1)]


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
    "like",
    "not like",
    "ilike",
    "not ilike",
    "in",
    "not in",
    "is null",
    "is not null",
    "AND",
    "OR",
]

VALUE_FILTER_OPERATORS = {
    "=",
    "eq",
    "equals",
    "!=",
    ">",
    ">=",
    "<",
    "<=",
    "like",
    "not like",
    "ilike",
    "not ilike",
}
SET_FILTER_OPERATORS = {"in", "not in"}
NULL_FILTER_OPERATORS = {"is null", "is not null"}
LOGICAL_FILTER_OPERATORS = {"AND", "OR"}


class SemanticFilter(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    operator: FilterOperator
    field: SemanticName | None = None
    value: Any = None
    values: tuple[Any, ...] | None = None
    conditions: tuple[SemanticFilter, ...] | None = None

    @model_validator(mode="after")
    def validate_operator_arguments(self) -> SemanticFilter:
        if self.operator in LOGICAL_FILTER_OPERATORS:
            if not self.conditions:
                raise ValueError("AND and OR filters require non-empty conditions.")
            if self.model_fields_set & {"field", "value", "values"}:
                raise ValueError(
                    "AND and OR filters accept conditions only, not field, value, or values."
                )
        elif self.operator in SET_FILTER_OPERATORS:
            if self.field is None or not self.values:
                raise ValueError(
                    "in and not in filters require a field and non-empty values."
                )
            if self.model_fields_set & {"value", "conditions"}:
                raise ValueError(
                    "in and not in filters accept values, not value or conditions."
                )
        elif self.operator in NULL_FILTER_OPERATORS:
            if self.field is None:
                raise ValueError("is null and is not null filters require a field.")
            if self.model_fields_set & {"value", "values", "conditions"}:
                raise ValueError(
                    "is null and is not null filters do not accept value, values, or conditions."
                )
        elif self.operator in VALUE_FILTER_OPERATORS:
            if self.field is None or "value" not in self.model_fields_set:
                raise ValueError("Comparison filters require both a field and a value.")
            if self.model_fields_set & {"values", "conditions"}:
                raise ValueError(
                    "Comparison filters accept value, not values or conditions."
                )
        return self

    def as_bsl_filter(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_unset=True)


class SemanticOrder(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    field: SemanticName
    direction: Literal["asc", "desc"] = "asc"


class SemanticTimeRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    start: str
    end: str

    @model_validator(mode="after")
    def validate_order(self) -> SemanticTimeRange:
        try:
            start = datetime.fromisoformat(self.start)
            end = datetime.fromisoformat(self.end)
        except ValueError as exc:
            raise ValueError("Time range bounds must be ISO 8601 timestamps.") from exc
        if start >= end:
            raise ValueError("Time range start must be before end.")
        return self


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
) -> TableOutput:
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

    return await create_table_output(
        ctx.deps.table_outputs,
        ctx.deps.spark,
        dataframe,
        error_prefix="Semantic query execution failed while reading the preview",
    )


@semantic_toolset.tool
def list_semantic_models(ctx: RunContext[Deps]) -> ListSemanticModelsResult:
    """List governed semantic models and their business descriptions.

    Use this before catalog or raw SQL for any possible governed metric, KPI,
    aggregate, or calculation. Do not assume a metric is absent without
    checking. Describe a plausible model next unless its exact fields are
    already known from tool results. Use only returned model names.
    """
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

    Use this before querying a plausible model unless its exact fields are
    already known from tool results. Model summaries and catalog columns are not
    semantic field schemas. Use only names returned by semantic tools.

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
    dimensions: SemanticNames | None = None,
    measures: SemanticNames | None = None,
    filters: list[SemanticFilter] | None = None,
    order_by: list[SemanticOrder] | None = None,
    limit: Annotated[int | None, Field(ge=1, le=100_000)] = None,
    time_grain: SemanticName | None = None,
    time_grains: dict[SemanticName, SemanticName] | None = None,
    time_range: SemanticTimeRange | None = None,
) -> TableOutput:
    """Query governed dimensions and measures from one semantic model.

    Use this when the model represents the request; never recreate, approximate,
    or verify its measures with raw SQL. Dimensions group or filter; measures
    calculate governed values. All field names must come from semantic
    discovery, while filter values may come from the user or data results.

    Prefer ``time_range`` for time restrictions (inclusive start, exclusive
    end). Use time grains only when different aggregation is requested.

    The result is saved with a preview: answer from it when sufficient, inspect
    it for more rows or columns, display it when the user requests the data, or
    pass it to the visualization tool for a chart.

    Args:
        model_name: Exact model name returned by ``list_semantic_models``.
        dimensions: Governed dimensions used to group the result.
        measures: Governed measures to calculate.
        filters: Dimension or measure filters. Use ``value`` for comparisons,
            ``values`` for ``in``/``not in``, and ``conditions`` for ``AND``/``OR``.
        order_by: Selected result fields and their sort directions.
        limit: Maximum result rows.
        time_grain: One grain applied to selected time dimensions.
        time_grains: Per-dimension grains; cannot be combined with time_grain.
        time_range: Inclusive start and exclusive end time range.
    """
    if not dimensions and not measures:
        raise ModelRetry("Select at least one dimension or measure.")
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
            time_range=time_range.model_dump(mode="json") if time_range else None,
        )
    except Exception as exc:
        raise ModelRetry(f"Semantic query is invalid: {exc}") from exc
    return await _execute_semantic_expression(ctx, expression)


@semantic_toolset.tool(retries=3, sequential=True)
async def compare_semantic_periods(
    ctx: RunContext[Deps],
    model_name: str,
    measures: SemanticNames,
    current_time_range: SemanticTimeRange,
    previous_time_range: SemanticTimeRange,
    dimensions: SemanticNames | None = None,
    filters: list[SemanticFilter] | None = None,
    time_dimension: SemanticName | None = None,
    time_grain: SemanticName | None = None,
    time_grains: dict[SemanticName, SemanticName] | None = None,
    order_by: list[SemanticOrder] | None = None,
    limit: Annotated[int | None, Field(ge=1, le=100_000)] = None,
) -> TableOutput:
    """Compare governed measures across two explicit time ranges.

    Use this for period-over-period questions instead of combining queries or
    raw SQL. Discover and describe the model first, then use only returned
    semantic names. The saved output contains current, previous, delta, and
    percentage-change columns.

    Args:
        model_name: Exact model name returned by ``list_semantic_models``.
        measures: Governed measures to compare.
        current_time_range: Inclusive start and exclusive end for the current period.
        previous_time_range: Inclusive start and exclusive end for the prior period.
        dimensions: Optional governed dimensions used to group the comparison.
        filters: Dimension or measure filters applied to both periods.
        time_dimension: Time dimension to use when the model has more than one.
        time_grain: One grain applied to selected time dimensions.
        time_grains: Per-dimension grains; cannot be combined with time_grain.
        order_by: Selected result fields and their sort directions.
        limit: Maximum result rows.
    """
    if not measures:
        raise ModelRetry("Select at least one measure to compare.")
    if time_grain is not None and time_grains is not None:
        raise ModelRetry("Use either time_grain or time_grains, not both.")
    model = _model(ctx, model_name)
    try:
        expression = model.compare_periods(
            dimensions=dimensions,
            measures=measures,
            current_time_range=current_time_range.model_dump(mode="json"),
            previous_time_range=previous_time_range.model_dump(mode="json"),
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

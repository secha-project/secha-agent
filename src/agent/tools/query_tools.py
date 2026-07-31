from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext, ToolReturn

from ..deps import Deps
from ..models import InspectQueryResult, QueryDataResult
from ..query_results import (
    UnknownQueryResultReferenceError,
    render_result_preview,
)
from ..runtime import run_spark_with_timeout
from ..spark import concise_spark_error, spark_error_type
from ..sql_safety import ReadOnlySqlError, require_read_only_sql


async def query_data(ctx: RunContext[Deps], sql: str) -> QueryDataResult:
    """Run Spark SQL and preview its first 10 rows and columns.

    Use this for questions about row data, calculations, aggregates, or data
    values. Do not use it to search catalog metadata. The lazy query result is
    saved and Spark validates the query before the preview is read.

    Args:
        sql: The Spark SQL query to execute.
    """
    deps = ctx.deps
    if not sql.strip():
        raise ModelRetry("SQL validation failed (EmptySql): SQL text is empty.")

    # Temporary application-layer protection until the catalog credentials enforce
    # read-only access. Remove this call/import and src.agent.sql_safety once those
    # permissions are the authoritative safeguard.
    try:
        require_read_only_sql(sql)
    except ReadOnlySqlError as exc:
        raise ModelRetry(
            f"SQL validation failed (ReadOnlyQueryRequired): {exc}"
        ) from exc

    try:
        result = deps.spark.sql(sql)
        result.schema
    except Exception as exc:
        raise ModelRetry(
            f"SQL validation failed ({spark_error_type(exc)}): "
            f"{concise_spark_error(exc)}. "
        ) from exc

    try:
        preview = await run_spark_with_timeout(
            deps.spark,
            lambda: render_result_preview(result),
        )
    except Exception as exc:
        raise ModelRetry(
            "SQL execution failed while reading the preview "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}. "
        ) from exc

    ref = deps.result_registry.create(result)
    return QueryDataResult(
        dataframe_ref=ref,
        preview_markdown=preview,
    )


async def inspect_query_result(
    ctx: RunContext[Deps],
    ref: str,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> InspectQueryResult:
    """Inspect rows and columns from result ``ref`` saved by ``query_data``.

    The row range is 1-based and inclusive. By default, this shows 10 rows and
    the first 10 columns. ``column_names`` selects specific columns instead.
    Row positions are only deterministic when the original SQL defines an order.

    Args:
        ref: A result reference available in the current live session.
        start_row: First row to show (1-based and inclusive).
        end_row: Last row to show (1-based and inclusive).
        column_names: Specific columns to show instead of the first 10.
    """
    deps = ctx.deps
    try:
        result = deps.result_registry.resolve(ref)
    except UnknownQueryResultReferenceError as exc:
        raise ModelRetry(
            f"Query-result reference {ref!r} is stale and no longer exists "
            "in memory. Call query_data again to create a new reference; "
            "do not retry with the stale reference."
        ) from exc
    if column_names:
        available_columns = result.schema.fieldNames()
        unknown_columns = [
            column_name
            for column_name in column_names
            if column_name not in available_columns
        ]
        if unknown_columns:
            raise ModelRetry(
                f"Unknown result columns: {', '.join(unknown_columns)}. "
                f"Available columns: {', '.join(available_columns)}."
            )

    try:
        preview = await run_spark_with_timeout(
            deps.spark,
            lambda: render_result_preview(
                result,
                start_row=start_row,
                end_row=end_row,
                column_names=column_names,
            ),
        )
    except Exception as exc:
        raise ModelRetry(
            "SQL execution failed while inspecting the result "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc
    return InspectQueryResult(
        dataframe_ref=ref,
        preview_markdown=preview,
    )


async def display_query_result(
    ctx: RunContext[Deps],
    ref: str,
) -> ToolReturn[str]:
    """Display a saved query result as a table in the user interface.

    Use this when the user requests data, rows, records, a table, or
    downloadable data. Do not use it for a scalar answer or explanation alone.
    Displaying the table does not replace the rest of the response: after
    it succeeds, continue with any explanation, interpretation, caveats, or
    other parts the user requested.

    Args:
        ref: A result reference available in the current live session.
    """
    deps = ctx.deps
    try:
        result = deps.result_registry.resolve(ref)
    except UnknownQueryResultReferenceError as exc:
        raise ModelRetry(
            f"Query-result reference {ref!r} is stale and no longer exists "
            "in memory. Call query_data again to create a new reference; "
            "do not retry with the stale reference."
        ) from exc

    if deps.result_exporter is not None:
        await deps.result_exporter(deps.spark, result)

    return ToolReturn(
        return_value=(
            f"Query result {ref} was displayed to the user as a table. "
            "Continue with any explanation or other requested parts of the answer. "
            "If nothing else was requested, briefly confirm the display."
        ),
        metadata={"ui_kind": "dataframe", "dataframe_ref": ref},
    )

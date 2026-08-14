from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.toolsets import FunctionToolset

from ..artifacts.base import (
    ArtifactRef,
    ArtifactSourceUnavailableError,
    UnexpectedArtifactSourceTypeError,
    UnknownArtifactError,
)
from ..artifacts.table import SparkTableSource
from ..deps import Deps
from ..models import InspectQueryResult, QueryDataResult
from ..spark import concise_spark_error, spark_error_type
from ..sql_safety import ReadOnlySqlError, require_read_only_sql

query_toolset = FunctionToolset[Deps]()


@query_toolset.tool(retries=3, sequential=True)
async def query_data(ctx: RunContext[Deps], sql: str) -> QueryDataResult:
    """Run Spark SQL and preview its first 10 rows and columns.

    Use this for questions about row data, calculations, aggregates, or data
    values. This tool does not display data to the user, if the user asks to see or
    download data, pass the reference this tool returns to display data using
    display_query_result.

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
        _ = result.schema
    except Exception as exc:
        raise ModelRetry(
            f"SQL validation failed ({spark_error_type(exc)}): "
            f"{concise_spark_error(exc)}. "
        ) from exc

    source = SparkTableSource(
        spark=deps.spark,
        dataframe=result,
    )
    try:
        preview = await source.preview()
    except Exception as exc:
        raise ModelRetry(
            "SQL execution failed while reading the preview "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}. "
        ) from exc

    ref = deps.artifact_context.create(source)
    return QueryDataResult(
        artifact_ref=ref,
        preview_markdown=preview.to_markdown(),
    )


@query_toolset.tool(sequential=True)
async def inspect_query_result(
    ctx: RunContext[Deps],
    ref: ArtifactRef,
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
        source = deps.artifact_context.resolve_source(ref, SparkTableSource)
    except (UnknownArtifactError, ArtifactSourceUnavailableError) as exc:
        raise ModelRetry(
            f"The live source for table artifact {ref.artifact_id!r} is no "
            "longer available. Call query_data again to create a new artifact; "
            "do not retry with this reference."
        ) from exc
    except UnexpectedArtifactSourceTypeError as exc:
        raise ModelRetry(str(exc)) from exc

    try:
        preview = await source.preview(
            start_row=start_row,
            end_row=end_row,
            column_names=column_names,
        )
    except Exception as exc:
        raise ModelRetry(
            "SQL execution failed while inspecting the result "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc
    return InspectQueryResult(
        artifact_ref=ref,
        preview_markdown=preview.to_markdown(),
    )


@query_toolset.tool(sequential=True)
async def display_query_result(
    ctx: RunContext[Deps],
    ref: ArtifactRef,
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
    if ref.artifact_kind != "table":
        raise ModelRetry(f"Artifact {ref.artifact_id!r} is not a table artifact.")
    try:
        artifact_ref = await deps.artifact_context.publish(ref)
    except (UnknownArtifactError, ArtifactSourceUnavailableError) as exc:
        raise ModelRetry(
            f"Table artifact {ref.artifact_id!r} is no longer available in "
            "memory. Call query_data again to create a new artifact; "
            "do not retry with this reference."
        ) from exc
    except TypeError as exc:
        raise ModelRetry(str(exc)) from exc

    return ToolReturn(
        return_value=(
            f"Table artifact {ref.artifact_id} was published. "
            "Continue with any explanation or other requested parts of the answer. "
            "If nothing else was requested, briefly confirm that it is available."
        ),
        metadata=artifact_ref,
    )

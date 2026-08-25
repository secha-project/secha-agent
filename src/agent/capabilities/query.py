from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type
from ..sql_safety import ReadOnlySqlError, require_read_only_sql
from ..table_result import TableResult, create_table_result

NAME = "query"
DESCRIPTION = "Run read-only SQL and produce saved table results."
query_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[query_toolset],
    )


@query_toolset.tool(retries=3, sequential=True)
async def query_data(ctx: RunContext[Deps], sql: str) -> TableResult:
    """Run read-only Spark SQL and save a table result with a small preview.

    Use this for row-level data, actual values, aggregates, and calculations not
    represented by a governed semantic measure. Resolve uncertain identifiers
    with catalog tools first; do not use schema-search tools for row values.

    For a possible governed metric, KPI, aggregate, or calculation, check
    semantic models and describe plausible matches first. Use raw SQL only when
    no model represents the calculation or row-level analysis is required.
    Never duplicate or validate a governed result with hand-written SQL.

    Answer from the preview when it is sufficient. Otherwise pass the returned
    reference to ``inspect_table_result``. This tool does not publish data in
    the UI: when the user asks to see or download rows, records, a table, sample
    data, or a sample time series, pass the reference to
    ``display_table_result`` instead of reproducing the result in Markdown.

    Args:
        sql: One complete read-only Spark SQL query. Data-changing statements
            and multiple statements are rejected.
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

    return await create_table_result(
        deps,
        result,
        error_prefix="SQL execution failed while reading the preview",
    )

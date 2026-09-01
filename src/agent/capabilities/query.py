from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..outputs.tables import TableOutput, create_table_output
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type
from ..sql_safety import ReadOnlySqlError, require_read_only_sql

NAME = "query"
DESCRIPTION = "Run read-only SQL and produce saved table outputs."
query_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[query_toolset],
    )


@query_toolset.tool(retries=3, sequential=True)
async def query_data(ctx: RunContext[Deps], sql: str) -> TableOutput:
    """Run read-only Spark SQL and save a table output with a small preview.

    Use this for row-level data, actual values, aggregates, and calculations.
    Resolve uncertain identifiers with catalog tools first; do not use
    schema-search tools for row values. Follow the active capability policy when
    choosing between governed objects and source-table calculations.

    Answer from the preview when it is sufficient. Otherwise pass the returned
    reference to ``inspect_table_output``. This tool does not publish data in
    the UI: when the user asks to see or download rows, records, a table, sample
    data, or a sample time series, pass the reference to
    ``display_table_output`` instead of reproducing the result in Markdown.

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

    return await create_table_output(
        deps.table_outputs,
        deps.spark,
        result,
        error_prefix="SQL execution failed while reading the preview",
    )

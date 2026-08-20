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
    """Run Spark SQL and preview its first 10 rows and columns.

    Use this for questions about row data, calculations, aggregates, or data
    values. This tool does not display data to the user. If the user asks to see
    or download data, pass its returned reference to ``display_table_result``.

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

    return await create_table_result(
        deps,
        result,
        error_prefix="SQL execution failed while reading the preview",
    )

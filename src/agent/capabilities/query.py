from __future__ import annotations

from collections.abc import Mapping

from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..outputs.tables import TableOutput, create_table_output
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type
from ..sql_safety import ReadOnlySqlError, require_read_only_sql

NAME = "query"
# This prompt will only be sent when using deferred loading of capabilities.
DESCRIPTION = "Run read-only SQL and produce saved table outputs."


def build(
    selected: frozenset[str],
    *,
    prompt_overrides: Mapping[str, str] | None = None,
) -> Capability[Deps]:
    toolset = FunctionToolset[Deps]()
    toolset.add_function(
        query_data,
        description=render_prompt(
            f"capabilities/{NAME}/tools/query_data.md",
            selected,
            overrides=prompt_overrides,
        ),
        retries=3,
        sequential=True,
    )
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(
            f"capabilities/{NAME}/instructions.md", selected, overrides=prompt_overrides
        ),
        toolsets=[toolset],
    )


async def query_data(ctx: RunContext[Deps], sql: str) -> TableOutput:
    """Run read-only Spark SQL and save a table output with a small preview.

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

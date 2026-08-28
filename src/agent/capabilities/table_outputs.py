from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..outputs.tables import TableOutput, TableOutputRef, UnknownTableOutputError
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type

NAME = "table_outputs"
DESCRIPTION = "Inspect or publish table outputs produced by data capabilities."
table_outputs_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[table_outputs_toolset],
    )


@table_outputs_toolset.tool(sequential=True)
async def inspect_table_output(
    ctx: RunContext[Deps],
    ref: TableOutputRef,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> TableOutput:
    """Inspect rows and columns from a saved table output.

    Use this when the producer's preview does not contain enough rows or columns
    to answer reliably. The row range is 1-based and inclusive. By default this
    shows 10 rows and the first 10 columns; ``column_names`` selects specific
    columns instead. Row positions are deterministic only when the producing
    query defines an order. Inspection does not publish the table to the user.

    Args:
        ref: A table-output reference available in the current live session.
        start_row: First row to show (1-based and inclusive).
        end_row: Last row to show (1-based and inclusive).
        column_names: Specific columns to show instead of the first 10.
    """
    deps = ctx.deps
    try:
        output = deps.table_outputs.resolve(ref)
    except UnknownTableOutputError as exc:
        raise ModelRetry(
            f"Table output {ref.output_id!r} is no longer available. "
            "Rerun the tool that produced it; do not retry with this reference."
        ) from exc

    try:
        preview = await output.preview(
            start_row=start_row,
            end_row=end_row,
            column_names=column_names,
        )
    except Exception as exc:
        raise ModelRetry(
            "Table-output inspection failed "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc
    return TableOutput(output_ref=ref, preview_markdown=preview.to_markdown())


@table_outputs_toolset.tool(sequential=True)
async def display_table_output(
    ctx: RunContext[Deps],
    ref: TableOutputRef,
) -> ToolReturn[str]:
    """Publish a saved table output in the user interface for viewing or download.

    Use this when the user asks to see, get, or download rows, records, tables,
    sample data, or a sample time series. Do not use it for a scalar answer or
    explanation alone, and do not manually reproduce the output as a Markdown
    or text table. Publishing does not replace requested explanation,
    interpretation, or caveats.

    Args:
        ref: A table-output reference available in the current live session.
    """
    deps = ctx.deps
    try:
        output = deps.table_outputs.resolve(ref)
        artifact = await output.to_artifact(deps.table_artifact_max_rows)
        artifact_ref = deps.artifacts.publish(artifact)
    except UnknownTableOutputError as exc:
        raise ModelRetry(
            f"Table output {ref.output_id!r} is no longer available. "
            "Rerun the tool that produced it; do not retry with this reference."
        ) from exc

    return ToolReturn(
        return_value=(
            f"Table artifact {artifact.id} was published. "
            "Continue with any explanation or other requested parts of the answer. "
            "If nothing else was requested, briefly confirm that it is available."
        ),
        metadata=artifact_ref,
    )

from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..deps import Deps
from ..outputs.tables import TableOutput, TableOutputRef, UnknownTableOutputError
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type

NAME = "table_outputs"
# This prompt will only be sent when using deferred loading of capabilities.
DESCRIPTION = "Inspect or publish table outputs produced by data capabilities."


def build(selected: frozenset[str]) -> Capability[Deps]:
    toolset = FunctionToolset[Deps]()
    toolset.add_function(
        inspect_table_output,
        description=render_prompt(
            f"capabilities/{NAME}/tools/inspect_table_output.md", selected
        ),
        sequential=True,
    )
    toolset.add_function(
        display_table_output,
        description=render_prompt(
            f"capabilities/{NAME}/tools/display_table_output.md", selected
        ),
        sequential=True,
    )
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[toolset],
    )


async def inspect_table_output(
    ctx: RunContext[Deps],
    ref: TableOutputRef,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> TableOutput:
    """Inspect rows and columns from a saved table output.

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


async def display_table_output(
    ctx: RunContext[Deps],
    ref: TableOutputRef,
) -> ToolReturn[str]:
    """Publish a saved table output in the user interface for viewing or download.

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

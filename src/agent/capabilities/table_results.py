from __future__ import annotations

from pydantic_ai import ModelRetry, RunContext, ToolReturn
from pydantic_ai.capabilities import Capability
from pydantic_ai.toolsets import FunctionToolset

from ..artifacts.base import (
    ArtifactRef,
    ArtifactSourceUnavailableError,
    UnexpectedArtifactSourceTypeError,
    UnknownArtifactError,
)
from ..artifacts.table import SparkTableSource
from ..deps import Deps
from ..prompt import render_prompt
from ..spark_runtime import concise_spark_error, spark_error_type
from ..table_result import TableResult

NAME = "table_results"
DESCRIPTION = "Inspect or publish table results produced by data capabilities."
table_results_toolset = FunctionToolset[Deps]()


def build(selected: frozenset[str]) -> Capability[Deps]:
    return Capability(
        id=NAME,
        description=DESCRIPTION,
        instructions=render_prompt(f"capabilities/{NAME}/instructions.md", selected),
        toolsets=[table_results_toolset],
    )


@table_results_toolset.tool(sequential=True)
async def inspect_table_result(
    ctx: RunContext[Deps],
    ref: ArtifactRef,
    start_row: int = 1,
    end_row: int | None = None,
    column_names: list[str] | None = None,
) -> TableResult:
    """Inspect rows and columns from a saved table result.

    The row range is 1-based and inclusive. By default, this shows 10 rows and
    the first 10 columns. ``column_names`` selects specific columns instead.
    Row positions are only deterministic when the producing query defines an
    order.

    Args:
        ref: A table-result reference available in the current live session.
        start_row: First row to show (1-based and inclusive).
        end_row: Last row to show (1-based and inclusive).
        column_names: Specific columns to show instead of the first 10.
    """
    deps = ctx.deps
    try:
        source = deps.artifacts.resolve_source(ref, SparkTableSource)
    except (UnknownArtifactError, ArtifactSourceUnavailableError) as exc:
        raise ModelRetry(
            f"The live source for table artifact {ref.artifact_id!r} is no "
            "longer available. Rerun the tool that produced this table result; "
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
            "Table-result inspection failed "
            f"({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc
    return TableResult(
        artifact_ref=ref,
        preview_markdown=preview.to_markdown(),
    )


@table_results_toolset.tool(sequential=True)
async def display_table_result(
    ctx: RunContext[Deps],
    ref: ArtifactRef,
) -> ToolReturn[str]:
    """Display a saved table result in the user interface.

    Use this when the user requests data, rows, records, a table, or
    downloadable data. Do not use it for a scalar answer or explanation alone.
    Displaying the table does not replace the rest of the response: after it
    succeeds, continue with any requested explanation, interpretation, or
    caveats.

    Args:
        ref: A table-result reference available in the current live session.
    """
    deps = ctx.deps
    try:
        artifact = deps.artifacts.registry.resolve(ref)
        if artifact.kind != "table":
            raise ModelRetry(f"Artifact {ref.artifact_id!r} is not a table artifact.")
        artifact_ref = await deps.artifacts.publish(ref)
    except (UnknownArtifactError, ArtifactSourceUnavailableError) as exc:
        raise ModelRetry(
            f"Table artifact {ref.artifact_id!r} is no longer available in "
            "memory. Rerun the tool that produced this table result; do not "
            "retry with this reference."
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

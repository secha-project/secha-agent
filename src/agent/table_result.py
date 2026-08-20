from __future__ import annotations

from pydantic import BaseModel
from pydantic_ai import ModelRetry
from pyspark.sql import DataFrame

from .artifacts.base import ArtifactRef
from .artifacts.table import SparkTableSource
from .deps import Deps
from .spark_runtime import concise_spark_error, spark_error_type


class TableResult(BaseModel):
    artifact_ref: ArtifactRef
    preview_markdown: str


async def create_table_result(
    deps: Deps,
    dataframe: DataFrame,
    *,
    error_prefix: str = "Table query failed while reading the preview",
) -> TableResult:
    """Register and preview a Spark DataFrame as a shared table result."""
    source = SparkTableSource(
        spark=deps.spark,
        dataframe=dataframe,
    )
    try:
        preview = await source.preview()
    except Exception as exc:
        raise ModelRetry(
            f"{error_prefix} ({spark_error_type(exc)}): {concise_spark_error(exc)}."
        ) from exc

    ref = deps.artifacts.create(source)
    return TableResult(
        artifact_ref=ref,
        preview_markdown=preview.to_markdown(),
    )

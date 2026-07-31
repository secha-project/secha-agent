from __future__ import annotations

from typing import Any

from pyspark.sql import SparkSession

from src.agent.query_results import QueryResultRegistry
from src.agent.runtime import run_spark_with_timeout

from .models import TableDisplayPart
from .table_artifacts import (
    MAX_DISPLAY_ROWS,
    TableSnapshot,
    arrow_bytes_to_table,
    snapshot_table,
    snapshot_to_csv,
)


class ResultDisplayHandler:
    """Turn query-result tool metadata into ordered, persistable UI output."""

    def __init__(
        self,
        spark: SparkSession,
        registry: QueryResultRegistry,
    ) -> None:
        self.spark = spark
        self.registry = registry
        self.snapshots: list[TableSnapshot] = []

    async def render_tool_result(
        self,
        metadata: object,
        container: Any,
    ) -> TableDisplayPart | None:
        if not isinstance(metadata, dict) or metadata.get("ui_kind") != "dataframe":
            return None

        ref = metadata.get("dataframe_ref")
        if not isinstance(ref, str):
            raise TypeError("The query-result tool returned invalid display metadata.")

        snapshot = await materialize_table_snapshot(
            self.spark,
            self.registry,
            ref,
        )
        self.snapshots.append(snapshot)
        render_snapshot(
            container,
            snapshot,
            key=f"download-live-{snapshot.id}",
        )
        return TableDisplayPart(artifact_id=snapshot.id)


async def materialize_table_snapshot(
    spark: SparkSession,
    registry: QueryResultRegistry,
    ref: str,
) -> TableSnapshot:
    """Materialize one extra row so the persisted snapshot records truncation."""
    result = registry.resolve(ref)
    materialized = await run_spark_with_timeout(
        spark,
        lambda: result.limit(MAX_DISPLAY_ROWS + 1).toPandas(),
    )
    return snapshot_table(materialized, max_rows=MAX_DISPLAY_ROWS)


def render_snapshot(
    container: Any,
    snapshot: TableSnapshot,
    *,
    key: str | None = None,
) -> None:
    """Render the exact persisted snapshot and a CSV download when truncated."""
    table = arrow_bytes_to_table(snapshot.arrow_bytes)
    container.dataframe(
        table.to_pandas(),
        width="stretch",
        hide_index=True,
    )
    if snapshot.truncated:
        container.caption(
            f"Showing the first {snapshot.visible_row_count:,} rows; "
            "the result was truncated."
        )
        container.download_button(
            "Download displayed rows as CSV",
            snapshot_to_csv(snapshot),
            file_name=f"secha-{snapshot.id}.csv",
            mime="text/csv",
            key=key,
        )

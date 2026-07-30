from __future__ import annotations

import re
from typing import Any, Protocol

from pydantic_ai import ModelRetry
from pyspark.sql import DataFrame, SparkSession

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


class QueryRecipeRepository(Protocol):
    def save_query_recipe(self, conversation_id: str, sql: str) -> int: ...

    def load_query_recipe(
        self,
        conversation_id: str,
        query_id: int,
    ) -> str | None: ...


class ConversationResultRegistry(QueryResultRegistry):
    """Conversation-scoped Spark results backed by durable SQL recipes."""

    _REFERENCE_PATTERN = re.compile(r"df-([1-9][0-9]*)")

    def __init__(
        self,
        spark: SparkSession,
        repository: QueryRecipeRepository,
        conversation_id: str,
    ) -> None:
        super().__init__()
        self.spark = spark
        self.repository = repository
        self.conversation_id = conversation_id

    def _allocate_reference(self, sql: str) -> str:
        query_id = self.repository.save_query_recipe(self.conversation_id, sql)
        return f"df-{query_id}"

    def _resolve_missing(self, ref: str) -> DataFrame:
        match = self._REFERENCE_PATTERN.fullmatch(ref)
        if match is None:
            raise ModelRetry(
                f"Invalid result reference {ref!r}. Use a reference such as 'df-17'."
            )

        sql = self.repository.load_query_recipe(
            self.conversation_id,
            int(match.group(1)),
        )
        if sql is None:
            raise ModelRetry(
                f"Result reference {ref!r} does not belong to this conversation."
            )

        try:
            result = self.spark.sql(sql)
        except Exception as exc:
            raise ModelRetry(
                f"Could not recreate result reference {ref!r}: {exc}"
            ) from exc
        return result


class ResultDisplayHandler:
    """Turn query-result tool metadata into ordered, persistable UI output."""

    def __init__(self, registry: ConversationResultRegistry) -> None:
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

        snapshot = await materialize_table_snapshot(self.registry, ref)
        self.snapshots.append(snapshot)
        render_snapshot(
            container,
            snapshot,
            key=f"download-live-{snapshot.id}",
        )
        return TableDisplayPart(artifact_id=snapshot.id)


async def materialize_table_snapshot(
    registry: ConversationResultRegistry,
    ref: str,
) -> TableSnapshot:
    """Materialize one extra row so the persisted snapshot records truncation."""
    result = registry.resolve(ref)
    materialized = await run_spark_with_timeout(
        registry.spark,
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

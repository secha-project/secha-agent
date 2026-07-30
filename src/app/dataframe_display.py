from __future__ import annotations

from typing import Any

from .dataframes import (
    MAX_DISPLAY_ROWS,
    DataframeArtifact,
    DataFrameReferences,
    arrow_bytes_to_table,
    artifact_to_csv,
    snapshot_table,
)
from .models import DataframeDisplayPart
from .runtime import run_spark_with_timeout


class DataframeDisplayHandler:
    """Turn dataframe tool metadata into ordered, persistable UI output."""

    def __init__(self, references: DataFrameReferences) -> None:
        self.references = references
        self.artifacts: list[DataframeArtifact] = []

    async def render_tool_result(
        self,
        metadata: object,
        container: Any,
    ) -> DataframeDisplayPart | None:
        if not isinstance(metadata, dict) or metadata.get("ui_kind") != "dataframe":
            return None

        ref = metadata.get("dataframe_ref")
        if not isinstance(ref, str):
            raise TypeError("The dataframe tool returned invalid display metadata.")

        artifact = await materialize_dataframe_artifact(self.references, ref)
        self.artifacts.append(artifact)
        render_artifact(
            container,
            artifact,
            key=f"download-live-{artifact.id}",
        )
        return DataframeDisplayPart(artifact_id=artifact.id)


async def materialize_dataframe_artifact(
    references: DataFrameReferences,
    ref: str,
) -> DataframeArtifact:
    """Materialize one extra row so the persisted snapshot records truncation."""
    dataframe = references.resolve(ref)
    materialized = await run_spark_with_timeout(
        references.spark,
        lambda: dataframe.limit(MAX_DISPLAY_ROWS + 1).toPandas(),
    )
    return snapshot_table(materialized, max_rows=MAX_DISPLAY_ROWS)


def render_artifact(
    container: Any,
    artifact: DataframeArtifact,
    *,
    key: str | None = None,
) -> None:
    """Render the exact persisted snapshot and a CSV download when truncated."""
    table = arrow_bytes_to_table(artifact.arrow_bytes)
    container.dataframe(
        table.to_pandas(),
        width="stretch",
        hide_index=True,
    )
    if artifact.truncated:
        container.caption(
            f"Showing the first {artifact.visible_row_count:,} rows; "
            "the result was truncated."
        )
        container.download_button(
            "Download displayed rows as CSV",
            artifact_to_csv(artifact),
            file_name=f"secha-{artifact.id}.csv",
            mime="text/csv",
            key=key,
        )

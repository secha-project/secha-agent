from __future__ import annotations

from typing import Any

import pandas as pd

from src.agent.outputs.artifacts import (
    Artifact,
    ArtifactCollector,
    ArtifactRef,
    ChartArtifact,
    TableArtifact,
)

from .models import ArtifactDisplayPart


class ArtifactRenderer:
    """Render live or persisted artifacts in the Streamlit application."""

    def render_tool_result(
        self,
        metadata: object,
        artifacts: ArtifactCollector,
        container: Any,
    ) -> ArtifactDisplayPart | None:
        if not isinstance(metadata, ArtifactRef):
            return None
        artifact = artifacts.resolve(metadata)
        self.render(
            container,
            artifact,
            key=f"download-live-{artifact.id}",
        )
        return ArtifactDisplayPart(artifact_id=artifact.id)

    def render(
        self,
        container: Any,
        artifact: Artifact,
        *,
        key: str | None = None,
    ) -> None:
        if isinstance(artifact, TableArtifact):
            self._render_table(
                container,
                artifact,
                key=key,
            )
            return
        if isinstance(artifact, ChartArtifact):
            self._render_chart(container, artifact)
            return
        raise TypeError(f"No renderer is registered for {type(artifact).__name__}.")

    def _render_table(
        self,
        container: Any,
        artifact: TableArtifact,
        *,
        key: str | None = None,
    ) -> None:
        """Render the exact persisted table and a CSV download when truncated."""
        container.dataframe(
            artifact.to_pandas(),
            width="stretch",
            hide_index=True,
        )
        if artifact.truncated:
            container.caption(
                f"Showing the first {artifact.row_count:,} rows; "
                "the result was truncated."
            )
            container.download_button(
                "Download displayed rows as CSV",
                artifact.to_csv(),
                file_name=f"secha-{artifact.id}.csv",
                mime="text/csv",
                key=key,
            )

    @staticmethod
    def _render_chart(
        container: Any,
        artifact: ChartArtifact,
    ) -> None:
        """Render the exact self-contained Vega-Lite chart."""
        spec = artifact.to_spec()
        data_reference = spec.pop("data", None)
        datasets = spec.pop("datasets", None)
        if (
            isinstance(data_reference, dict)
            and isinstance(data_reference.get("name"), str)
            and isinstance(datasets, dict)
            and data_reference["name"] in datasets
        ):
            container.vega_lite_chart(
                pd.DataFrame(datasets[data_reference["name"]]),
                spec,
                width="stretch",
                height=500,
                theme=None,
            )
            return

        if data_reference is not None:
            spec["data"] = data_reference
        if datasets is not None:
            spec["datasets"] = datasets
        container.vega_lite_chart(
            spec,
            width="stretch",
            height=500,
            theme=None,
        )

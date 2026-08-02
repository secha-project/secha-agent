from __future__ import annotations

from typing import Any

from src.agent.artifacts.base import (
    Artifact,
    ArtifactRef,
    ArtifactRegistry,
)
from src.agent.artifacts.table import TableContent

from .models import ArtifactDisplayPart


class ArtifactRenderer:
    """Render live or persisted artifacts in the Streamlit application."""

    def __init__(self, registry: ArtifactRegistry) -> None:
        self.registry = registry

    def render_tool_result(
        self,
        metadata: object,
        container: Any,
    ) -> ArtifactDisplayPart | None:
        if not isinstance(metadata, ArtifactRef):
            return None

        artifact = self.registry.resolve(metadata)
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
        content = artifact.require_content()
        if isinstance(content, TableContent):
            self._render_table(
                container,
                artifact,
                content,
                key=key,
            )
            return
        raise TypeError(
            f"No renderer is registered for {artifact.kind!r} artifacts."
        )

    def _render_table(
        self,
        container: Any,
        artifact: Artifact,
        content: TableContent,
        *,
        key: str | None = None,
    ) -> None:
        """Render the exact persisted table and a CSV download when truncated."""
        container.dataframe(
            content.to_pandas(),
            width="stretch",
            hide_index=True,
        )
        if content.truncated:
            container.caption(
                f"Showing the first {content.row_count:,} rows; "
                "the result was truncated."
            )
            container.download_button(
                "Download displayed rows as CSV",
                content.to_csv(),
                file_name=f"secha-{artifact.id}.csv",
                mime="text/csv",
                key=key,
            )

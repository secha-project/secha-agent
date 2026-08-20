from __future__ import annotations

import streamlit as st

from ..artifact_renderer import ArtifactRenderer
from ..models import DisplayMessage
from ..resources import AppResources
from ..storage.history import ArtifactNotFoundError
from ..streaming import render_tool


def render_message(
    message: DisplayMessage,
    *,
    resources: AppResources,
    artifact_renderer: ArtifactRenderer,
    user_id: str,
    conversation_id: str,
) -> None:
    # Session-state objects can survive a hot reload and retain their previous
    # Python class identity. Discriminator fields remain stable across reloads.
    if message.role == "user":
        st.chat_message("user").markdown(message.content)
        return

    container = st.chat_message("assistant")
    for part in message.parts:
        if part.kind == "text":
            container.markdown(part.content)
        elif part.kind == "tool":
            render_tool(container, part)
        elif part.kind == "artifact":
            try:
                artifact = resources.history.load_artifact(
                    user_id,
                    conversation_id,
                    part.artifact_id,
                )
            except ArtifactNotFoundError:
                container.error("This saved artifact is unavailable.")
            else:
                artifact_renderer.render(
                    container,
                    artifact,
                    key=f"download-history-{part.artifact_id}",
                )

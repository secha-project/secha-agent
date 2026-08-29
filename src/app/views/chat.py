from __future__ import annotations

import asyncio

import streamlit as st

from src.agent.outputs.tables import TableOutputStore

from ..models import DisplayMessage, User
from ..rendering import render_artifact, render_tool
from ..resources import AppResources
from ..state import create_conversation, ensure_active_conversation
from ..storage.history import ArtifactNotFoundError
from ..streaming import stream_agent_response
from .sidebar import render_sidebar


def render_message(
    message: DisplayMessage,
    *,
    resources: AppResources,
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
                render_artifact(
                    container,
                    artifact,
                    key=f"download-history-{part.artifact_id}",
                )


def _table_outputs(conversation_id: str | None) -> TableOutputStore | None:
    if conversation_id is None:
        return None
    outputs = st.session_state.table_outputs
    if not isinstance(outputs, TableOutputStore):
        raise TypeError("The active conversation has no table output store.")
    return outputs


def _render_prompt_input(
    resources: AppResources,
    user: User,
    conversation_id: str | None,
    table_outputs: TableOutputStore | None,
) -> str | None:
    if conversation_id is None:
        with st.container(
            key="empty-chat-state",
            horizontal_alignment="center",
            vertical_alignment="center",
            gap="medium",
        ):
            st.subheader(
                "Ask questions about SECHA's data platform",
                text_alignment="center",
            )
            return st.chat_input(
                "Ask SECHA Agent",
                key="chat-prompt",
                submit_mode="stop",
                on_submit=create_conversation,
                args=(resources, user.id),
                width=720,
            )

    if table_outputs is None:
        raise RuntimeError("The active conversation has no table output store.")
    for message in st.session_state.display_messages:
        render_message(
            message,
            resources=resources,
            user_id=user.id,
            conversation_id=conversation_id,
        )
    return st.chat_input(
        "Ask SECHA Agent",
        key="chat-prompt",
        submit_mode="stop",
    )


def render_chat(resources: AppResources, user: User) -> None:
    conversation_id = ensure_active_conversation(resources, user.id)
    table_outputs = _table_outputs(conversation_id)
    render_sidebar(resources, user)
    prompt = _render_prompt_input(
        resources,
        user,
        conversation_id,
        table_outputs,
    )
    if not prompt:
        return

    if conversation_id is None:
        conversation_id = st.session_state.current_conversation_id
    if table_outputs is None:
        table_outputs = _table_outputs(conversation_id)
    if table_outputs is None:
        raise RuntimeError("The submitted prompt has no table output store.")
    saved = resources.history.append_user(user.id, conversation_id, prompt)
    st.session_state.display_messages = saved.display_messages
    st.session_state.model_messages = saved.model_messages
    render_message(
        saved.display_messages[-1],
        resources=resources,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    container = st.chat_message("assistant")
    run = resources.agent.start_run(
        prompt=prompt,
        message_history=st.session_state.model_messages,
        table_outputs=table_outputs,
    )
    try:
        result = asyncio.run(
            stream_agent_response(
                events=run.events,
                artifacts=run.artifacts,
                container=container,
            )
        )
    except Exception as exc:  # noqa: BLE001 - UI boundary reports all failures.
        container.error(f"Unable to complete the request. Retry when ready: {exc}")
        return

    if result is None:
        return

    completed = resources.history.commit_assistant(
        user.id,
        conversation_id,
        result.assistant_message,
        result.model_messages,
        result.artifacts,
    )
    st.session_state.display_messages = completed.display_messages
    st.session_state.model_messages = completed.model_messages
    st.rerun()

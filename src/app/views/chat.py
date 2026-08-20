from __future__ import annotations

import asyncio

import streamlit as st
from pydantic_ai import ModelSettings, UsageLimits

from src.agent.artifacts.base import ArtifactRegistry

from ..artifact_renderer import ArtifactRenderer
from ..models import User
from ..resources import AppResources, fresh_deps
from ..state import create_conversation, ensure_active_conversation
from ..streaming import stream_agent_response
from .messages import render_message
from .sidebar import render_sidebar


def _artifact_renderer(conversation_id: str | None) -> ArtifactRenderer | None:
    if conversation_id is None:
        return None
    registry = st.session_state.artifact_registry
    if not isinstance(registry, ArtifactRegistry):
        raise TypeError("The active conversation has no artifact registry.")
    return ArtifactRenderer(registry)


def _render_prompt_input(
    resources: AppResources,
    user: User,
    conversation_id: str | None,
    artifact_renderer: ArtifactRenderer | None,
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

    if artifact_renderer is None:
        raise RuntimeError("The active conversation has no artifact renderer.")
    for message in st.session_state.display_messages:
        render_message(
            message,
            resources=resources,
            artifact_renderer=artifact_renderer,
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
    artifact_renderer = _artifact_renderer(conversation_id)
    render_sidebar(resources, user)
    prompt = _render_prompt_input(
        resources,
        user,
        conversation_id,
        artifact_renderer,
    )
    if not prompt:
        return

    if conversation_id is None:
        conversation_id = st.session_state.current_conversation_id
    if artifact_renderer is None:
        artifact_renderer = _artifact_renderer(conversation_id)
    if artifact_renderer is None:
        raise RuntimeError("The submitted prompt has no artifact renderer.")

    saved = resources.history.append_user(user.id, conversation_id, prompt)
    st.session_state.display_messages = saved.display_messages
    st.session_state.model_messages = saved.model_messages
    render_message(
        saved.display_messages[-1],
        resources=resources,
        artifact_renderer=artifact_renderer,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    container = st.chat_message("assistant")
    deps = fresh_deps(resources, artifact_renderer.registry)
    try:
        result = asyncio.run(
            stream_agent_response(
                prompt=prompt,
                message_history=st.session_state.model_messages,
                deps=deps,
                agent_instance=resources.agent,
                model=resources.config.create_model(),
                model_settings=ModelSettings(temperature=resources.config.temperature),
                usage_limits=UsageLimits(request_limit=resources.config.request_limit),
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

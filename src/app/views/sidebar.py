from __future__ import annotations

import streamlit as st

from ..models import User
from ..resources import AppResources
from ..routes.auth import AUTH_LOGOUT_ENDPOINT
from ..state import (
    SIDEBAR_HISTORY_REVISION_KEY,
    activate_conversation,
    auth_cookie_token,
    clear_authenticated_session,
    delete_conversation,
    start_new_conversation,
)
from .authentication import render_auth_request

CONVERSATION_BUTTON_LABEL_MAX_LENGTH = 26


def conversation_button_label(title: str) -> str:
    if len(title) <= CONVERSATION_BUTTON_LABEL_MAX_LENGTH:
        return title
    return title[: CONVERSATION_BUTTON_LABEL_MAX_LENGTH - 1] + "…"


def render_sidebar(resources: AppResources, user: User) -> None:
    with st.sidebar:
        with st.container(key="sidebar-header", gap="xsmall"):
            st.header("SECHA Agent")
            st.button(
                "New chat",
                icon=":material/add:",
                on_click=start_new_conversation,
                width="stretch",
            )
        history_revision = st.session_state.get(SIDEBAR_HISTORY_REVISION_KEY, 0)
        with st.container(
            key=f"sidebar-history-{history_revision}",
            gap="xsmall",
        ):
            st.subheader("Chat history")
            conversations = resources.history.list_conversations(user.id)
            if not conversations:
                st.caption("Your conversations will appear here.")
            for conversation in conversations:
                with st.container(
                    key=f"conversation-row-{conversation.id}",
                    gap=None,
                ):
                    st.button(
                        conversation_button_label(conversation.title),
                        key=f"conversation-{conversation.id}",
                        type=(
                            "primary"
                            if conversation.id
                            == st.session_state.current_conversation_id
                            else "secondary"
                        ),
                        on_click=activate_conversation,
                        args=(resources, user.id, conversation.id),
                        width="stretch",
                    )
                    st.button(
                        "Delete conversation",
                        key=f"delete-conversation-{conversation.id}",
                        icon=":material/delete:",
                        type="tertiary",
                        on_click=delete_conversation,
                        args=(resources, user.id, conversation.id),
                    )
        with st.container(key="sidebar-account", gap="xsmall"):
            st.caption("Signed in as")
            with st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
                gap="small",
            ):
                st.markdown(f":material/account_circle: **{user.username}**")
                if st.button(
                    "Sign out",
                    icon=":material/logout:",
                    type="tertiary",
                    width="content",
                ):
                    token = auth_cookie_token()
                    if token is not None:
                        resources.history.revoke_auth_session(token)
                    clear_authenticated_session()
                    render_auth_request(AUTH_LOGOUT_ENDPOINT)

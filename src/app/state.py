from __future__ import annotations

import streamlit as st

from src.agent.outputs.tables import TableOutputStore

from .models import User
from .resources import AppResources
from .routes.auth import AUTH_COOKIE_NAME
from .storage.history import ConversationNotFoundError, History

AUTHENTICATED_USER_KEY = "authenticated_user_id"
SIDEBAR_HISTORY_REVISION_KEY = "sidebar_history_revision"
APP_VIEW_KEY = "app_view"
AUTH_SESSION_KEYS = (
    AUTHENTICATED_USER_KEY,
    "current_conversation_id",
    "display_messages",
    "model_messages",
    "table_outputs",
    SIDEBAR_HISTORY_REVISION_KEY,
    APP_VIEW_KEY,
)


def set_authenticated_user(user: User) -> None:
    st.session_state[AUTHENTICATED_USER_KEY] = user.id


def auth_cookie_token() -> str | None:
    token = st.context.cookies.get(AUTH_COOKIE_NAME)
    return token if isinstance(token, str) else None


def clear_authenticated_session() -> None:
    for key in AUTH_SESSION_KEYS:
        st.session_state.pop(key, None)
    st.query_params.clear()


def current_user(history: History) -> User | None:
    user_id = st.session_state.get(AUTHENTICATED_USER_KEY)
    if isinstance(user_id, str):
        user = history.get_user(user_id)
        if user is None:
            clear_authenticated_session()
        return user

    token = auth_cookie_token()
    if token is None:
        return None
    user = history.get_user_by_auth_session(token)
    if user is not None:
        set_authenticated_user(user)
    return user


def activate_conversation(
    resources: AppResources,
    user_id: str,
    conversation_id: str,
) -> bool:
    try:
        conversation = resources.history.load_conversation(
            user_id,
            conversation_id,
        )
    except ConversationNotFoundError:
        return False

    st.session_state[APP_VIEW_KEY] = "chat"
    st.session_state.current_conversation_id = conversation.id
    st.session_state.display_messages = conversation.display_messages
    st.session_state.model_messages = conversation.model_messages
    st.session_state.table_outputs = TableOutputStore()
    st.query_params["chat"] = conversation.id
    return True


def start_new_conversation() -> None:
    st.session_state[APP_VIEW_KEY] = "chat"
    st.session_state.current_conversation_id = None
    st.session_state.display_messages = []
    st.session_state.model_messages = []
    st.session_state.table_outputs = None
    st.query_params.pop("chat", None)


def show_metric_views() -> None:
    st.session_state[APP_VIEW_KEY] = "metric_views"
    st.query_params.pop("chat", None)
    st.session_state["metric-view-mode"] = "list"
    st.session_state.pop("metric-view-draft", None)
    st.session_state.pop("metric-view-new-dimensions", None)
    st.session_state.pop("metric-view-new-measures", None)
    st.session_state.pop("metric-view-error", None)
    st.session_state.pop("metric-view-tabs", None)


def create_conversation(resources: AppResources, user_id: str) -> None:
    conversation = resources.history.create_conversation(user_id)
    activate_conversation(resources, user_id, conversation.id)


def delete_conversation(
    resources: AppResources,
    user_id: str,
    conversation_id: str,
) -> None:
    try:
        resources.history.delete_conversation(user_id, conversation_id)
    except ConversationNotFoundError:
        return

    st.session_state[SIDEBAR_HISTORY_REVISION_KEY] = (
        st.session_state.get(SIDEBAR_HISTORY_REVISION_KEY, 0) + 1
    )
    if conversation_id != st.session_state.get("current_conversation_id"):
        return

    conversations = resources.history.list_conversations(user_id)
    if conversations:
        activate_conversation(resources, user_id, conversations[0].id)
    else:
        start_new_conversation()


def ensure_active_conversation(
    resources: AppResources,
    user_id: str,
) -> str | None:
    requested_id = st.query_params.get("chat")
    current_id = st.session_state.get("current_conversation_id")
    table_outputs = st.session_state.get("table_outputs")

    if requested_id:
        requested_id = str(requested_id)
        state_is_loaded = (
            requested_id == current_id
            and isinstance(st.session_state.get("display_messages"), list)
            and isinstance(st.session_state.get("model_messages"), list)
            and isinstance(table_outputs, TableOutputStore)
        )
        if not state_is_loaded:
            if activate_conversation(resources, user_id, requested_id):
                return requested_id
            del st.query_params["chat"]
        else:
            return requested_id

    if "current_conversation_id" in st.session_state:
        if current_id is None:
            st.session_state.setdefault("display_messages", [])
            st.session_state.setdefault("model_messages", [])
            st.session_state["table_outputs"] = None
            return None
        if isinstance(current_id, str):
            state_is_loaded = (
                isinstance(st.session_state.get("display_messages"), list)
                and isinstance(st.session_state.get("model_messages"), list)
                and isinstance(table_outputs, TableOutputStore)
            )
            if state_is_loaded or activate_conversation(resources, user_id, current_id):
                return current_id

    start_new_conversation()
    return None

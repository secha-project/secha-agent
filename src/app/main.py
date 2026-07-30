from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass

import logfire
import streamlit as st
from pydantic_ai import ModelSettings, UsageLimits
from pyspark.sql import SparkSession

from src.agent.agent import agent
from src.agent.catalog import UnityCatalogClient
from src.agent.config import AgentConfig
from src.agent.deps import Deps
from src.agent.providers import get_model
from src.agent.spark import get_spark_session
from src.app.models import (
    DisplayMessage,
    User,
)
from src.app.query_results import ConversationResultRegistry, render_snapshot
from src.app.storage import (
    ArtifactNotFoundError,
    ChatRepository,
    ConversationNotFoundError,
    InvalidUsernameError,
    UsernameAlreadyExistsError,
    WeakPasswordError,
)
from src.app.streaming import render_tool, stream_agent_response

APP_MODEL_PROVIDER = "openai"
AUTHENTICATED_USER_KEY = "authenticated_user_id"
AUTH_SESSION_KEYS = (
    AUTHENTICATED_USER_KEY,
    "current_conversation_id",
    "display_messages",
    "model_messages",
    "result_registry",
)

APP_STYLES = """
<style>
header {
    background: transparent !important;
}
[data-testid="stAppDeployButton"],
#MainMenu {
    display: none;
}
[data-testid="stMainBlockContainer"] {
    padding-top: 0 !important;
}
[data-testid^="stChatMessageAvatar"] {
    display: none;
}
[data-testid="stChatMessage"]:has(
        [data-testid="stChatMessageAvatarUser"]
    ) {
    margin-left: auto;
    max-width: 75%;
    width: fit-content;
}
@media (max-width: 640px) {
    [data-testid="stChatMessage"]:has(
            [data-testid="stChatMessageAvatarUser"]
        ) {
        max-width: 90%;
    }
}
div[class*="st-key-empty-chat-state"] {
    min-height: calc(100dvh - 4rem);
}
[data-testid="stSidebarUserContent"] {
    /* Reclaim most of Streamlit's sidebar header without covering its
       native collapse control (which occupies the top-right 44px). */
    box-sizing: border-box;
    height: calc(100dvh - 48px);
    margin-top: -28px !important;
    padding-bottom: 0.75rem;
}
[data-testid="stSidebarUserContent"] > div,
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    height: 100%;
    min-height: 0;
}
[data-testid="stSidebarUserContent"] > div > [data-testid="stVerticalBlock"] {
    gap: 0.75rem;
}
[data-testid="stSidebarUserContent"]
    > div
    > [data-testid="stVerticalBlock"]
    > [data-testid="stLayoutWrapper"]:has(
        > div[class*="st-key-sidebar-history"]
    ) {
    flex: 1 1 auto;
    min-height: 0;
    overflow: hidden;
}
div[class*="st-key-sidebar-history"] {
    height: 100%;
    min-height: 0;
    overflow-y: auto;
    padding-right: 0.25rem;
}
div[class*="st-key-sidebar-account"] {
    background: color-mix(in srgb, currentColor 4%, transparent);
    border: 1px solid color-mix(in srgb, currentColor 12%, transparent);
    border-radius: 0.75rem;
    box-shadow: 0 1px 2px color-mix(in srgb, #000 8%, transparent);
    flex: 0 0 auto !important;
    padding: 0.75rem;
}
div[class*="st-key-sidebar-account"] [data-testid="stCaptionContainer"] p {
    margin-bottom: 0;
}
div[class*="st-key-sidebar-header"] {
    gap: 1.25rem;
}
div[class*="st-key-sidebar-header"] h2 {
    font-size: 1.5rem;
    line-height: 1.2;
    padding-block: 0.5rem;
    transform: translateY(-1rem);
}
div[class*="st-key-conversation-row-"] {
    position: relative;
}
div[class*="st-key-conversation-row-"] div[class*="st-key-conversation-"] button {
    padding-right: 2.75rem;
}
div[class*="st-key-conversation-row-"]
    div[class*="st-key-delete-conversation-"] {
    opacity: 0;
    position: absolute;
    right: 0.35rem;
    top: 0.3rem;
    transition: opacity 120ms ease-in-out;
    z-index: 1;
}
div[class*="st-key-conversation-row-"]:hover
    div[class*="st-key-delete-conversation-"],
div[class*="st-key-delete-conversation-"]:focus-within {
    opacity: 1;
}
div[class*="st-key-delete-conversation-"] button {
    border-radius: 0.35rem;
    height: 1.9rem;
    min-height: 1.9rem;
    padding: 0;
    width: 1.9rem;
}
div[class*="st-key-delete-conversation-"] button p {
    display: none;
}
@media (hover: none) {
    div[class*="st-key-delete-conversation-"] {
        opacity: 1 !important;
    }
}
</style>
"""


@dataclass(frozen=True)
class AppResources:
    config: AgentConfig
    spark: SparkSession
    catalog: UnityCatalogClient
    repository: ChatRepository


@st.cache_resource
def load_repository() -> ChatRepository:
    repository = ChatRepository()
    repository.initialize()
    return repository


@st.cache_resource
def load_resources() -> AppResources:
    config = AgentConfig(model_provider=APP_MODEL_PROVIDER)  # type: ignore[call-arg]
    os.environ["OTEL_EXPORTER_OTLP_ENDPOINT"] = config.otel_exporter_otlp_endpoint
    logfire.configure(
        send_to_logfire=False,
        service_name=config.logfire_service_name,
        metrics=False,
    )
    logfire.instrument_pydantic_ai()

    return AppResources(
        config=config,
        spark=get_spark_session(config),
        catalog=UnityCatalogClient(
            config.catalog_url,
            config.catalog_token,
            config.catalog_name,
        ),
        repository=load_repository(),
    )


def _set_authenticated_user(user: User) -> None:
    st.session_state[AUTHENTICATED_USER_KEY] = user.id


def _clear_authenticated_session() -> None:
    for key in AUTH_SESSION_KEYS:
        st.session_state.pop(key, None)
    st.query_params.clear()


def _current_user(repository: ChatRepository) -> User | None:
    user_id = st.session_state.get(AUTHENTICATED_USER_KEY)
    if not isinstance(user_id, str):
        return None
    user = repository.get_user(user_id)
    if user is None:
        _clear_authenticated_session()
    return user


def _render_authentication(repository: ChatRepository) -> None:
    st.title("SECHA")
    st.caption("Sign in to continue to your analytics assistant.")
    mode = st.segmented_control(
        "Account action",
        ["Log in", "Sign up"],
        default="Log in",
        required=True,
        key="auth-mode",
        label_visibility="collapsed",
        width="stretch",
    )

    with st.container(border=True):
        if mode == "Sign up":
            st.subheader("Create an account")
            with st.form("signup-form", border=False):
                username = st.text_input(
                    "Username",
                    max_chars=32,
                    autocomplete="username",
                    icon=":material/person:",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    max_chars=128,
                    autocomplete="new-password",
                    help="Use at least 8 characters.",
                    icon=":material/lock:",
                )
                confirmation = st.text_input(
                    "Confirm password",
                    type="password",
                    max_chars=128,
                    autocomplete="new-password",
                    icon=":material/lock:",
                )
                submitted = st.form_submit_button(
                    "Create account",
                    type="primary",
                    icon=":material/person_add:",
                    width="stretch",
                )
            if not submitted:
                return
            if password != confirmation:
                st.error("Passwords do not match.")
                return
            try:
                user = repository.create_user(username, password)
            except (
                InvalidUsernameError,
                WeakPasswordError,
                UsernameAlreadyExistsError,
            ) as exc:
                st.error(str(exc))
                return
            _set_authenticated_user(user)
            st.rerun()

        st.subheader("Log in")
        with st.form("login-form", border=False):
            username = st.text_input(
                "Username",
                max_chars=32,
                autocomplete="username",
                icon=":material/person:",
            )
            password = st.text_input(
                "Password",
                type="password",
                max_chars=128,
                autocomplete="current-password",
                icon=":material/lock:",
            )
            submitted = st.form_submit_button(
                "Log in",
                type="primary",
                icon=":material/login:",
                width="stretch",
            )
        if not submitted:
            return
        user = repository.authenticate_user(username, password)
        if user is None:
            st.error("Invalid username or password.")
            return
        _set_authenticated_user(user)
        st.rerun()


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
        elif part.kind == "dataframe":
            try:
                snapshot = resources.repository.load_artifact(
                    user_id,
                    conversation_id,
                    part.artifact_id,
                )
            except ArtifactNotFoundError:
                container.error("This saved table snapshot is unavailable.")
            else:
                render_snapshot(
                    container,
                    snapshot,
                    key=f"download-history-{part.artifact_id}",
                )


def activate_conversation(
    resources: AppResources,
    user_id: str,
    conversation_id: str,
) -> bool:
    try:
        conversation = resources.repository.load_conversation(
            user_id,
            conversation_id,
        )
    except ConversationNotFoundError:
        return False

    st.session_state.current_conversation_id = conversation.id
    st.session_state.display_messages = conversation.display_messages
    st.session_state.model_messages = conversation.model_messages
    st.session_state.result_registry = ConversationResultRegistry(
        resources.spark,
        resources.repository,
        conversation.id,
    )
    st.query_params["chat"] = conversation.id
    return True


def start_new_conversation() -> None:
    """Reset the UI to an unsaved draft conversation."""
    st.session_state.current_conversation_id = None
    st.session_state.display_messages = []
    st.session_state.model_messages = []
    st.session_state.result_registry = None
    if "chat" in st.query_params:
        del st.query_params["chat"]


def create_conversation(
    resources: AppResources,
    user_id: str,
) -> None:
    """Create and select a conversation before its first prompt is handled."""
    conversation = resources.repository.create_conversation(user_id)
    activate_conversation(resources, user_id, conversation.id)


def delete_conversation(
    resources: AppResources,
    user_id: str,
    conversation_id: str,
) -> None:
    try:
        resources.repository.delete_conversation(
            user_id,
            conversation_id,
        )
    except ConversationNotFoundError:
        return

    if conversation_id != st.session_state.get("current_conversation_id"):
        return

    conversations = resources.repository.list_conversations(user_id)
    if conversations:
        activate_conversation(resources, user_id, conversations[0].id)
    else:
        start_new_conversation()


def _ensure_active_conversation(
    resources: AppResources,
    user_id: str,
) -> str | None:
    requested_id = st.query_params.get("chat")
    current_id = st.session_state.get("current_conversation_id")
    registry = st.session_state.get("result_registry")

    if requested_id:
        requested_id = str(requested_id)
        if requested_id != current_id or (
            not isinstance(registry, ConversationResultRegistry)
            or registry.conversation_id != current_id
        ):
            if activate_conversation(resources, user_id, requested_id):
                return requested_id
            del st.query_params["chat"]

    if "current_conversation_id" in st.session_state:
        if current_id is None:
            st.session_state.setdefault("display_messages", [])
            st.session_state.setdefault("model_messages", [])
            st.session_state["result_registry"] = None
            return None
        if isinstance(current_id, str):
            if (
                isinstance(registry, ConversationResultRegistry)
                and registry.conversation_id == current_id
            ) or activate_conversation(
                resources,
                user_id,
                current_id,
            ):
                return current_id

    conversations = resources.repository.list_conversations(user_id)
    if conversations:
        fallback_id = conversations[0].id
        if activate_conversation(resources, user_id, fallback_id):
            return fallback_id

    start_new_conversation()
    return None


def _render_sidebar(
    resources: AppResources,
    user: User,
) -> None:
    with st.sidebar:
        with st.container(key="sidebar-header", gap="xsmall"):
            st.header("SECHA Agent")
            st.button(
                "New chat",
                icon=":material/add:",
                on_click=start_new_conversation,
                width="stretch",
            )
        with st.container(key="sidebar-history", gap="xsmall"):
            st.subheader("Chat history")
            conversations = resources.repository.list_conversations(user.id)
            if not conversations:
                st.caption("Your conversations will appear here.")
            for conversation in conversations:
                with st.container(
                    key=f"conversation-row-{conversation.id}",
                    gap=None,
                ):
                    st.button(
                        conversation.title,
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
                        help=f'Delete "{conversation.title}"',
                        icon=":material/delete:",
                        type="tertiary",
                        on_click=delete_conversation,
                        args=(resources, user.id, conversation.id),
                    )
        with st.container(
            key="sidebar-account",
            gap="xsmall",
        ):
            st.caption("Signed in as")
            with st.container(
                horizontal=True,
                horizontal_alignment="distribute",
                vertical_alignment="center",
                gap="small",
            ):
                st.markdown(f":material/account_circle: **{user.username}**")
                st.button(
                    "Sign out",
                    icon=":material/logout:",
                    type="tertiary",
                    on_click=_clear_authenticated_session,
                    width="content",
                )


def _fresh_deps(
    resources: AppResources,
    registry: ConversationResultRegistry,
) -> Deps:
    return Deps(
        spark=resources.spark,
        catalog=resources.catalog,
        schema_name=resources.config.schema_name,
        table_column_page_size=resources.config.table_column_page_size,
        result_registry=registry,
    )


def run_app() -> None:
    st.set_page_config(page_title="SECHA", page_icon="🗃️")
    st.markdown(APP_STYLES, unsafe_allow_html=True)
    if "workspace" in st.query_params:
        del st.query_params["workspace"]

    repository = load_repository()
    user = _current_user(repository)
    if user is None:
        _render_authentication(repository)
        return

    resources = load_resources()
    conversation_id = _ensure_active_conversation(resources, user.id)

    _render_sidebar(resources, user)

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
            prompt = st.chat_input(
                "Ask SECHA Agent",
                key="chat-prompt",
                submit_mode="stop",
                on_submit=create_conversation,
                args=(resources, user.id),
                width=720,
            )
    else:
        for message in st.session_state.display_messages:
            render_message(
                message,
                resources=resources,
                user_id=user.id,
                conversation_id=conversation_id,
            )
        prompt = st.chat_input(
            "Ask SECHA Agent",
            key="chat-prompt",
            submit_mode="stop",
        )
    if not prompt:
        return

    if conversation_id is None:
        conversation_id = st.session_state.current_conversation_id

    # First save: the submitted user message is durable before agent work begins.
    saved = resources.repository.append_user(
        user.id,
        conversation_id,
        prompt,
    )
    st.session_state.display_messages = saved.display_messages
    st.session_state.model_messages = saved.model_messages
    render_message(
        saved.display_messages[-1],
        resources=resources,
        user_id=user.id,
        conversation_id=conversation_id,
    )

    container = st.chat_message("assistant")
    registry: ConversationResultRegistry = st.session_state.result_registry
    deps = _fresh_deps(resources, registry)
    try:
        result = asyncio.run(
            stream_agent_response(
                prompt=prompt,
                message_history=st.session_state.model_messages,
                deps=deps,
                agent_instance=agent,
                model=get_model(resources.config),
                model_settings=ModelSettings(temperature=resources.config.temperature),
                usage_limits=UsageLimits(request_limit=resources.config.request_limit),
                container=container,
            )
        )
    except Exception as exc:
        # The user message remains saved, but partial assistant output and model
        # history deliberately do not become conversation state.
        container.error(f"Unable to complete the request. Retry when ready: {exc}")
        return

    if result is None:
        return

    # Second save happens only for the terminal AgentRunResultEvent.
    completed = resources.repository.commit_assistant(
        user.id,
        conversation_id,
        result.assistant_message,
        result.model_messages,
        result.snapshots,
    )
    st.session_state.display_messages = completed.display_messages
    st.session_state.model_messages = completed.model_messages
    st.rerun()


if __name__ == "__main__":
    run_app()

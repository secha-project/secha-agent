from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.app.state as state_module
from src.agent.artifacts.base import (
    ArtifactRegistry,
    UnknownArtifactError,
)
from src.app.resources import fresh_deps
from src.app.routes.auth import AUTH_COOKIE_NAME
from src.app.views.sidebar import (
    CONVERSATION_BUTTON_LABEL_MAX_LENGTH,
    conversation_button_label,
)


class SessionState(dict[str, Any]):
    def __getattr__(self, key: str) -> Any:
        return self[key]

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value


def app_resources() -> SimpleNamespace:
    conversation = SimpleNamespace(
        id="conversation-1",
        display_messages=[],
        model_messages=[],
    )
    return SimpleNamespace(
        history=SimpleNamespace(
            load_conversation=lambda _user_id, _conversation_id: conversation,
        ),
        spark=object(),
        catalog=object(),
        semantic=None,
        config=SimpleNamespace(
            schema_name="bronze",
            table_column_page_size=20,
        ),
    )


def test_conversation_button_label_respects_maximum_length() -> None:
    title = "Show all measurements for the Helsinki facility"
    label = conversation_button_label(title)

    assert label == "Show all measurements for…"
    assert len(label) == CONVERSATION_BUTTON_LABEL_MAX_LENGTH


def test_short_conversation_button_label_is_unchanged() -> None:
    assert conversation_button_label("Show measurements") == "Show measurements"


def test_current_user_is_restored_from_auth_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState()
    user = SimpleNamespace(id="user-1")
    fake_streamlit = SimpleNamespace(
        session_state=state,
        query_params={},
        context=SimpleNamespace(cookies={AUTH_COOKIE_NAME: "session-token"}),
    )
    history = SimpleNamespace(
        get_user_by_auth_session=lambda token: (
            user if token == "session-token" else None
        )
    )
    monkeypatch.setattr(state_module, "st", fake_streamlit)

    assert state_module.current_user(cast(Any, history)) is user
    assert state[state_module.AUTHENTICATED_USER_KEY] == user.id


def test_active_conversation_reuses_its_artifact_registry_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState()
    fake_streamlit = SimpleNamespace(session_state=state, query_params={})
    monkeypatch.setattr(state_module, "st", fake_streamlit)
    resources = app_resources()

    assert state_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    registry = state.artifact_registry
    ref = registry.create(cast(Any, SimpleNamespace(kind="table")))

    assert (
        state_module.ensure_active_conversation(
            cast(Any, resources),
            "user-1",
        )
        == "conversation-1"
    )
    first_turn = fresh_deps(cast(Any, resources), registry)
    second_turn = fresh_deps(cast(Any, resources), registry)

    assert first_turn.artifacts.registry is registry
    assert second_turn.artifacts.registry.resolve(ref) is registry.resolve(ref)
    assert first_turn.artifacts is not second_turn.artifacts


def test_reloading_a_conversation_discards_live_references_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState()
    fake_streamlit = SimpleNamespace(session_state=state, query_params={})
    monkeypatch.setattr(state_module, "st", fake_streamlit)
    resources = app_resources()

    state_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    old_registry: ArtifactRegistry = state.artifact_registry
    old_ref = old_registry.create(cast(Any, SimpleNamespace(kind="table")))

    state_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    new_registry: ArtifactRegistry = state.artifact_registry

    assert new_registry is not old_registry
    with pytest.raises(UnknownArtifactError):
        new_registry.resolve(old_ref)


def test_deleting_inactive_conversation_refreshes_history(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    deleted: list[tuple[str, str]] = []
    state = SessionState(
        current_conversation_id="conversation-1",
        sidebar_history_revision=2,
    )
    fake_streamlit = SimpleNamespace(session_state=state, query_params={})
    resources = SimpleNamespace(
        history=SimpleNamespace(
            delete_conversation=lambda user_id, conversation_id: deleted.append(
                (user_id, conversation_id)
            ),
        )
    )
    monkeypatch.setattr(state_module, "st", fake_streamlit)

    state_module.delete_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-2",
    )

    assert deleted == [("user-1", "conversation-2")]
    assert state.sidebar_history_revision == 3
    assert state.current_conversation_id == "conversation-1"


def test_deleting_active_conversation_selects_remaining_conversation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replacement = SimpleNamespace(
        id="conversation-2",
        display_messages=[],
        model_messages=[],
    )
    state = SessionState(current_conversation_id="conversation-1")
    fake_streamlit = SimpleNamespace(session_state=state, query_params={})
    resources = SimpleNamespace(
        history=SimpleNamespace(
            delete_conversation=lambda _user_id, _conversation_id: None,
            list_conversations=lambda _user_id: [SimpleNamespace(id=replacement.id)],
            load_conversation=lambda _user_id, _conversation_id: replacement,
        )
    )
    monkeypatch.setattr(state_module, "st", fake_streamlit)

    state_module.delete_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )

    assert state.sidebar_history_revision == 1
    assert state.current_conversation_id == "conversation-2"
    assert fake_streamlit.query_params["chat"] == "conversation-2"

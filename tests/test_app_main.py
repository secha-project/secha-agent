from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.app.main as app_module
from src.agent.artifacts.base import (
    ArtifactRegistry,
    UnknownArtifactError,
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
        repository=SimpleNamespace(
            load_conversation=lambda _user_id, _conversation_id: conversation,
        ),
        spark=object(),
        catalog=object(),
        config=SimpleNamespace(
            schema_name="bronze",
            table_column_page_size=20,
        ),
    )


def test_conversation_button_label_respects_maximum_length() -> None:
    title = "Show all measurements for the Helsinki facility"
    label = app_module._conversation_button_label(title)

    assert label == "Show all measurements for…"
    assert len(label) == app_module.CONVERSATION_BUTTON_LABEL_MAX_LENGTH


def test_short_conversation_button_label_is_unchanged() -> None:
    assert app_module._conversation_button_label("Show measurements") == (
        "Show measurements"
    )


def test_current_user_is_restored_from_auth_cookie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState()
    user = SimpleNamespace(id="user-1")
    fake_streamlit = SimpleNamespace(
        session_state=state,
        query_params={},
        context=SimpleNamespace(cookies={app_module.AUTH_COOKIE_NAME: "session-token"}),
    )
    repository = SimpleNamespace(
        get_user_by_auth_session=lambda token: (
            user if token == "session-token" else None
        )
    )
    monkeypatch.setattr(app_module, "st", fake_streamlit)

    assert app_module._current_user(cast(Any, repository)) is user
    assert state[app_module.AUTHENTICATED_USER_KEY] == user.id


def test_active_conversation_reuses_its_artifact_registry_across_turns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState()
    fake_streamlit = SimpleNamespace(session_state=state, query_params={})
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    resources = app_resources()

    assert app_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    registry = state.artifact_registry
    ref = registry.create(cast(Any, SimpleNamespace(kind="table")))

    assert (
        app_module._ensure_active_conversation(
            cast(Any, resources),
            "user-1",
        )
        == "conversation-1"
    )
    first_turn = app_module._fresh_deps(cast(Any, resources), registry)
    second_turn = app_module._fresh_deps(cast(Any, resources), registry)

    assert first_turn.artifact_context.registry is registry
    assert second_turn.artifact_context.registry.resolve(ref) is registry.resolve(ref)
    assert first_turn.artifact_context is not second_turn.artifact_context


def test_reloading_a_conversation_discards_live_references_without_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = SessionState()
    fake_streamlit = SimpleNamespace(session_state=state, query_params={})
    monkeypatch.setattr(app_module, "st", fake_streamlit)
    resources = app_resources()

    app_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    old_registry: ArtifactRegistry = state.artifact_registry
    old_ref = old_registry.create(cast(Any, SimpleNamespace(kind="table")))

    app_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    new_registry: ArtifactRegistry = state.artifact_registry

    assert new_registry is not old_registry
    with pytest.raises(UnknownArtifactError):
        new_registry.resolve(old_ref)

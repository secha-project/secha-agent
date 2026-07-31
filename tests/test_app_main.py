from types import SimpleNamespace
from typing import Any, cast

import pytest

import src.app.main as app_module
from src.agent.query_results import (
    QueryResultRegistry,
    UnknownQueryResultReferenceError,
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

    assert label == "Show all measuremen…"
    assert len(label) == app_module.CONVERSATION_BUTTON_LABEL_MAX_LENGTH


def test_short_conversation_button_label_is_unchanged() -> None:
    assert app_module._conversation_button_label("Show measurements") == (
        "Show measurements"
    )


def test_active_conversation_reuses_its_live_registry_across_turns(
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
    registry = state.result_registry
    ref = registry.create(cast(Any, object()))

    assert app_module._ensure_active_conversation(
        cast(Any, resources),
        "user-1",
    ) == "conversation-1"
    first_turn = app_module._fresh_deps(cast(Any, resources), registry)
    second_turn = app_module._fresh_deps(cast(Any, resources), registry)

    assert first_turn.result_registry is registry
    assert second_turn.result_registry.resolve(ref) is registry.resolve(ref)


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
    old_registry: QueryResultRegistry = state.result_registry
    old_ref = old_registry.create(cast(Any, object()))

    app_module.activate_conversation(
        cast(Any, resources),
        "user-1",
        "conversation-1",
    )
    new_registry: QueryResultRegistry = state.result_registry

    assert new_registry is not old_registry
    with pytest.raises(UnknownQueryResultReferenceError):
        new_registry.resolve(old_ref)

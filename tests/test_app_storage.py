import json
import sqlite3
import stat
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic_ai import ModelMessage, ModelRequest, UserPromptPart

from src.agent.artifacts.base import Artifact, ArtifactRef
from src.agent.artifacts.visualization import VisualizationContent
from src.app.models import AssistantDisplayMessage, TextDisplayPart, User
from src.app.storage.history import (
    ArtifactNotFoundError,
    ConversationNotFoundError,
    History,
    InvalidUsernameError,
    UsernameAlreadyExistsError,
    WeakPasswordError,
)


@pytest.fixture
def history(tmp_path: Path) -> History:
    value = History(tmp_path / "chat.sqlite3")
    value.initialize()
    return value


@pytest.fixture
def users(history: History) -> tuple[User, User]:
    return (
        history.create_user("Alice", "correct-horse"),
        history.create_user("Bob", "battery-staple"),
    )


def test_users_are_registered_and_authenticated_case_insensitively(
    history: History,
) -> None:
    alice = history.create_user("Alice", "correct-horse")

    assert history.authenticate_user("alice", "correct-horse") == alice
    assert history.authenticate_user(" ALICE ", "correct-horse") == alice
    assert history.authenticate_user("alice", "wrong-password") is None
    assert history.authenticate_user("missing", "correct-horse") is None

    with sqlite3.connect(history.db_path) as connection:
        stored_hash = connection.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (alice.id,),
        ).fetchone()[0]
    assert stored_hash.startswith("$argon2id$")
    assert "correct-horse" not in stored_hash
    assert stat.S_IMODE(history.db_path.stat().st_mode) == 0o600


def test_registration_validates_credentials_and_unique_usernames(
    history: History,
) -> None:
    history.create_user("Alice", "correct-horse")

    with pytest.raises(UsernameAlreadyExistsError):
        history.create_user("alice", "another-password")
    with pytest.raises(InvalidUsernameError):
        history.create_user("no spaces allowed", "correct-horse")
    with pytest.raises(WeakPasswordError):
        history.create_user("Charlie", "short")


def test_auth_sessions_are_hashed_and_can_be_revoked(
    history: History,
) -> None:
    alice = history.create_user("Alice", "correct-horse")
    token = history.create_auth_session(alice.id)

    assert history.get_user_by_auth_session(token) == alice
    assert history.get_user_by_auth_session("not-a-session-token") is None

    with sqlite3.connect(history.db_path) as connection:
        stored_hash = connection.execute(
            "SELECT token_hash FROM auth_sessions WHERE user_id = ?",
            (alice.id,),
        ).fetchone()[0]
    assert stored_hash == sha256(token.encode("ascii")).hexdigest()
    assert token != stored_hash

    history.revoke_auth_session(token)
    assert history.get_user_by_auth_session(token) is None


def test_expired_auth_sessions_are_rejected(
    history: History,
) -> None:
    alice = history.create_user("Alice", "correct-horse")
    token = history.create_auth_session(alice.id)
    expired_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(history.db_path) as connection:
        connection.execute(
            "UPDATE auth_sessions SET expires_at = ?",
            (expired_at,),
        )

    assert history.get_user_by_auth_session(token) is None


def test_user_ownership_for_conversations_and_artifacts(
    history: History,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, bob = users
    alpha = history.create_conversation(alice.id)

    with pytest.raises(ConversationNotFoundError):
        history.load_conversation(bob.id, alpha.id)

    artifact = make_table_artifact([{"value": 1}])
    history.save_artifact(alice.id, alpha.id, artifact)
    with pytest.raises(ArtifactNotFoundError):
        history.load_artifact(bob.id, alpha.id, artifact.id)


def test_user_and_completed_assistant_are_saved_in_two_steps(
    history: History,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, _ = users
    conversation = history.create_conversation(alice.id)
    after_user = history.append_user(
        alice.id,
        conversation.id,
        "Show measurements",
    )

    assert [message.role for message in after_user.display_messages] == ["user"]
    assert after_user.model_messages == []

    model_history: list[ModelMessage] = [
        ModelRequest(parts=[UserPromptPart(content="Show measurements")])
    ]
    artifact = make_table_artifact([{"measurement": 1}])
    completed = history.commit_assistant(
        alice.id,
        conversation.id,
        AssistantDisplayMessage(
            parts=[TextDisplayPart(content="Here are the measurements.")]
        ),
        model_history,
        [artifact],
    )

    assert [message.role for message in completed.display_messages] == [
        "user",
        "assistant",
    ]
    assert len(completed.model_messages) == 1
    assert history.load_artifact(alice.id, conversation.id, artifact.id) == artifact


def test_conversation_appears_in_history_only_after_first_user_message(
    history: History,
    users: tuple[User, User],
) -> None:
    alice, _ = users
    conversation = history.create_conversation(alice.id)

    assert history.list_conversations(alice.id) == []

    history.append_user(alice.id, conversation.id, "Show measurements")

    summaries = history.list_conversations(alice.id)
    assert [summary.id for summary in summaries] == [conversation.id]
    assert summaries[0].title == "Show measurements"


def test_assistant_and_artifacts_are_committed_atomically(
    history: History,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, _ = users
    conversation = history.create_conversation(alice.id)
    history.append_user(alice.id, conversation.id, "Show measurements")
    existing_artifact = make_table_artifact([{"measurement": 1}])
    history.save_artifact(
        alice.id,
        conversation.id,
        existing_artifact,
    )
    conflicting_artifact = make_table_artifact(
        [{"measurement": 2}],
        artifact_id=existing_artifact.id,
    )

    with pytest.raises(sqlite3.IntegrityError):
        history.commit_assistant(
            alice.id,
            conversation.id,
            AssistantDisplayMessage(
                parts=[TextDisplayPart(content="Here are the measurements.")]
            ),
            [ModelRequest(parts=[UserPromptPart(content="Show measurements")])],
            [conflicting_artifact],
        )

    restored = history.load_conversation(alice.id, conversation.id)
    assert [message.role for message in restored.display_messages] == ["user"]
    assert restored.model_messages == []


def test_persisting_the_same_logical_artifact_again_is_idempotent(
    history: History,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, _ = users
    conversation = history.create_conversation(alice.id)
    artifact = make_table_artifact([{"measurement": 1}])
    history.save_artifact(alice.id, conversation.id, artifact)

    history.save_artifact(alice.id, conversation.id, artifact)

    assert (
        history.load_artifact(
            alice.id,
            conversation.id,
            artifact.id,
        )
        == artifact
    )


def test_delete_conversation_is_user_scoped_and_cascades(
    history: History,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, bob = users
    conversation = history.create_conversation(alice.id)
    artifact = make_table_artifact([{"value": 1}])
    history.save_artifact(
        alice.id,
        conversation.id,
        artifact,
    )

    with pytest.raises(ConversationNotFoundError):
        history.delete_conversation(bob.id, conversation.id)

    history.delete_conversation(alice.id, conversation.id)

    with pytest.raises(ConversationNotFoundError):
        history.load_conversation(alice.id, conversation.id)
    with pytest.raises(ArtifactNotFoundError):
        history.load_artifact(
            alice.id,
            conversation.id,
            artifact.id,
        )


def test_visualization_round_trip_persists_only_vega_lite_spec(
    history: History,
    users: tuple[User, User],
) -> None:
    alice, _ = users
    conversation = history.create_conversation(alice.id)
    spec = {"data": {"values": [{"x": 1}]}, "mark": "point"}
    artifact = Artifact(
        ref=ArtifactRef(artifact_id="visualization-1"),
        content=VisualizationContent(spec=spec),
    )

    history.save_artifact(alice.id, conversation.id, artifact)
    restored = history.load_artifact(
        alice.id,
        conversation.id,
        artifact.id,
    )

    restored_content = restored.require_content()
    assert isinstance(restored_content, VisualizationContent)
    assert restored_content.to_spec() == spec
    with sqlite3.connect(history.db_path) as connection:
        payload = connection.execute(
            "SELECT payload FROM artifacts WHERE id = ?",
            (artifact.id,),
        ).fetchone()[0]
    assert json.loads(payload) == spec

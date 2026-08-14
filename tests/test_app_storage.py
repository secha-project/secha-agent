import json
import sqlite3
import stat
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path

import pytest
from pydantic_ai import ModelRequest, UserPromptPart

from src.agent.artifacts.base import Artifact, ArtifactRef
from src.agent.artifacts.visualization import VisualizationContent
from src.app.models import AssistantDisplayMessage, TextDisplayPart, User
from src.app.storage import (
    ArtifactNotFoundError,
    ChatRepository,
    ConversationNotFoundError,
    InvalidUsernameError,
    UsernameAlreadyExistsError,
    WeakPasswordError,
)


@pytest.fixture
def repository(tmp_path: Path) -> ChatRepository:
    value = ChatRepository(tmp_path / "chat.sqlite3")
    value.initialize()
    return value


@pytest.fixture
def users(repository: ChatRepository) -> tuple[User, User]:
    return (
        repository.create_user("Alice", "correct-horse"),
        repository.create_user("Bob", "battery-staple"),
    )


def test_users_are_registered_and_authenticated_case_insensitively(
    repository: ChatRepository,
) -> None:
    alice = repository.create_user("Alice", "correct-horse")

    assert repository.authenticate_user("alice", "correct-horse") == alice
    assert repository.authenticate_user(" ALICE ", "correct-horse") == alice
    assert repository.authenticate_user("alice", "wrong-password") is None
    assert repository.authenticate_user("missing", "correct-horse") is None

    with sqlite3.connect(repository.db_path) as connection:
        stored_hash = connection.execute(
            "SELECT password_hash FROM users WHERE id = ?",
            (alice.id,),
        ).fetchone()[0]
    assert stored_hash.startswith("$argon2id$")
    assert "correct-horse" not in stored_hash
    assert stat.S_IMODE(repository.db_path.stat().st_mode) == 0o600


def test_registration_validates_credentials_and_unique_usernames(
    repository: ChatRepository,
) -> None:
    repository.create_user("Alice", "correct-horse")

    with pytest.raises(UsernameAlreadyExistsError):
        repository.create_user("alice", "another-password")
    with pytest.raises(InvalidUsernameError):
        repository.create_user("no spaces allowed", "correct-horse")
    with pytest.raises(WeakPasswordError):
        repository.create_user("Charlie", "short")


def test_auth_sessions_are_hashed_and_can_be_revoked(
    repository: ChatRepository,
) -> None:
    alice = repository.create_user("Alice", "correct-horse")
    token = repository.create_auth_session(alice.id)

    assert repository.get_user_by_auth_session(token) == alice
    assert repository.get_user_by_auth_session("not-a-session-token") is None

    with sqlite3.connect(repository.db_path) as connection:
        stored_hash = connection.execute(
            "SELECT token_hash FROM auth_sessions WHERE user_id = ?",
            (alice.id,),
        ).fetchone()[0]
    assert stored_hash == sha256(token.encode("ascii")).hexdigest()
    assert token != stored_hash

    repository.revoke_auth_session(token)
    assert repository.get_user_by_auth_session(token) is None


def test_expired_auth_sessions_are_rejected(
    repository: ChatRepository,
) -> None:
    alice = repository.create_user("Alice", "correct-horse")
    token = repository.create_auth_session(alice.id)
    expired_at = (datetime.now(UTC) - timedelta(seconds=1)).isoformat()
    with sqlite3.connect(repository.db_path) as connection:
        connection.execute(
            "UPDATE auth_sessions SET expires_at = ?",
            (expired_at,),
        )

    assert repository.get_user_by_auth_session(token) is None


def test_user_ownership_for_conversations_and_artifacts(
    repository: ChatRepository,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, bob = users
    alpha = repository.create_conversation(alice.id)

    with pytest.raises(ConversationNotFoundError):
        repository.load_conversation(bob.id, alpha.id)

    artifact = make_table_artifact([{"value": 1}])
    repository.save_artifact(alice.id, alpha.id, artifact)
    with pytest.raises(ArtifactNotFoundError):
        repository.load_artifact(bob.id, alpha.id, artifact.id)


def test_user_and_completed_assistant_are_saved_in_two_steps(
    repository: ChatRepository,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, _ = users
    conversation = repository.create_conversation(alice.id)
    after_user = repository.append_user(
        alice.id,
        conversation.id,
        "Show measurements",
    )

    assert [message.role for message in after_user.display_messages] == ["user"]
    assert after_user.model_messages == []

    model_history = [ModelRequest(parts=[UserPromptPart(content="Show measurements")])]
    artifact = make_table_artifact([{"measurement": 1}])
    completed = repository.commit_assistant(
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
    assert repository.load_artifact(alice.id, conversation.id, artifact.id) == artifact


def test_conversation_appears_in_history_only_after_first_user_message(
    repository: ChatRepository,
    users: tuple[User, User],
) -> None:
    alice, _ = users
    conversation = repository.create_conversation(alice.id)

    assert repository.list_conversations(alice.id) == []

    repository.append_user(alice.id, conversation.id, "Show measurements")

    summaries = repository.list_conversations(alice.id)
    assert [summary.id for summary in summaries] == [conversation.id]
    assert summaries[0].title == "Show measurements"


def test_assistant_and_artifacts_are_committed_atomically(
    repository: ChatRepository,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, _ = users
    conversation = repository.create_conversation(alice.id)
    repository.append_user(alice.id, conversation.id, "Show measurements")
    existing_artifact = make_table_artifact([{"measurement": 1}])
    repository.save_artifact(
        alice.id,
        conversation.id,
        existing_artifact,
    )
    conflicting_artifact = make_table_artifact(
        [{"measurement": 2}],
        artifact_id=existing_artifact.id,
    )

    with pytest.raises(sqlite3.IntegrityError):
        repository.commit_assistant(
            alice.id,
            conversation.id,
            AssistantDisplayMessage(
                parts=[TextDisplayPart(content="Here are the measurements.")]
            ),
            [ModelRequest(parts=[UserPromptPart(content="Show measurements")])],
            [conflicting_artifact],
        )

    restored = repository.load_conversation(alice.id, conversation.id)
    assert [message.role for message in restored.display_messages] == ["user"]
    assert restored.model_messages == []


def test_persisting_the_same_logical_artifact_again_is_idempotent(
    repository: ChatRepository,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, _ = users
    conversation = repository.create_conversation(alice.id)
    artifact = make_table_artifact([{"measurement": 1}])
    repository.save_artifact(alice.id, conversation.id, artifact)

    repository.save_artifact(alice.id, conversation.id, artifact)

    assert (
        repository.load_artifact(
            alice.id,
            conversation.id,
            artifact.id,
        )
        == artifact
    )


def test_delete_conversation_is_user_scoped_and_cascades(
    repository: ChatRepository,
    users: tuple[User, User],
    make_table_artifact,
) -> None:
    alice, bob = users
    conversation = repository.create_conversation(alice.id)
    artifact = make_table_artifact([{"value": 1}])
    repository.save_artifact(
        alice.id,
        conversation.id,
        artifact,
    )

    with pytest.raises(ConversationNotFoundError):
        repository.delete_conversation(bob.id, conversation.id)

    repository.delete_conversation(alice.id, conversation.id)

    with pytest.raises(ConversationNotFoundError):
        repository.load_conversation(alice.id, conversation.id)
    with pytest.raises(ArtifactNotFoundError):
        repository.load_artifact(
            alice.id,
            conversation.id,
            artifact.id,
        )


def test_visualization_round_trip_persists_only_vega_lite_spec(
    repository: ChatRepository,
    users: tuple[User, User],
) -> None:
    alice, _ = users
    conversation = repository.create_conversation(alice.id)
    spec_json = json.dumps({"data": {"values": [{"x": 1}]}, "mark": "point"}).encode()
    artifact = Artifact(
        ref=ArtifactRef(
            artifact_id="visualization-1",
            artifact_kind="visualization",
        ),
        content=VisualizationContent(spec_json=spec_json),
    )

    repository.save_artifact(alice.id, conversation.id, artifact)
    restored = repository.load_artifact(
        alice.id,
        conversation.id,
        artifact.id,
    )

    restored_content = restored.require_content()
    assert isinstance(restored_content, VisualizationContent)
    assert restored_content.spec_json == spec_json
    with sqlite3.connect(repository.db_path) as connection:
        payload = connection.execute(
            "SELECT payload FROM artifacts WHERE id = ?",
            (artifact.id,),
        ).fetchone()[0]
    assert bytes(payload) == spec_json

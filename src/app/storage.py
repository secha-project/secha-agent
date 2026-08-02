from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter

from src.agent.artifacts.base import (
    Artifact,
    ArtifactRef,
)
from src.agent.artifacts.table import ARROW_STREAM_MEDIA_TYPE, TableContent

from .models import (
    DISPLAY_MESSAGES_ADAPTER,
    AssistantDisplayMessage,
    Conversation,
    ConversationSummary,
    DisplayMessage,
    User,
    UserDisplayMessage,
)

DEFAULT_DB_PATH = Path(".data/secha_chat.sqlite3")
MIN_PASSWORD_LENGTH = 8
MAX_PASSWORD_LENGTH = 128
AUTH_SESSION_TOKEN_BYTES = 32
AUTH_SESSION_TOKEN_LENGTH = 43
AUTH_SESSION_TTL = timedelta(days=7)
AUTH_SESSION_TTL_SECONDS = int(AUTH_SESSION_TTL.total_seconds())
USERNAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{2,31}$")
PASSWORD_HASHER = PasswordHasher()


class ConversationNotFoundError(LookupError):
    pass


class ArtifactNotFoundError(LookupError):
    pass


class InvalidUsernameError(ValueError):
    pass


class WeakPasswordError(ValueError):
    pass


class UsernameAlreadyExistsError(ValueError):
    pass


def normalize_username(username: str) -> str:
    return username.strip().casefold()


def validate_username(username: str) -> str:
    cleaned = username.strip()
    if not USERNAME_PATTERN.fullmatch(cleaned):
        raise InvalidUsernameError(
            "Username must be 3–32 characters and use only letters, numbers, "
            "periods, underscores, or hyphens."
        )
    return cleaned


def validate_password(password: str) -> None:
    if len(password) < MIN_PASSWORD_LENGTH:
        raise WeakPasswordError(
            f"Password must be at least {MIN_PASSWORD_LENGTH} characters."
        )
    if len(password) > MAX_PASSWORD_LENGTH:
        raise WeakPasswordError(
            f"Password must be at most {MAX_PASSWORD_LENGTH} characters."
        )


def conversation_title(messages: Iterable[DisplayMessage]) -> str:
    for message in messages:
        if isinstance(message, UserDisplayMessage):
            title = " ".join(message.content.split())
            return title if len(title) <= 42 else f"{title[:39]}..."
    return "New conversation"


class ChatRepository:
    """Short-lived SQLite operations with user ownership checks."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        configured = db_path or os.environ.get("SECHA_CHAT_HISTORY_DB")
        self.db_path = Path(configured) if configured else DEFAULT_DB_PATH

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 5000")
        return connection

    def initialize(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY,
                    username TEXT NOT NULL,
                    normalized_username TEXT NOT NULL UNIQUE,
                    password_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    display_messages_json TEXT NOT NULL,
                    model_messages_json BLOB NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    FOREIGN KEY (user_id)
                        REFERENCES users (id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS conversations_user_updated_idx
                    ON conversations (user_id, updated_at DESC);

                CREATE TABLE IF NOT EXISTS auth_sessions (
                    token_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    FOREIGN KEY (user_id)
                        REFERENCES users (id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS auth_sessions_user_idx
                    ON auth_sessions (user_id, expires_at);

                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    media_type TEXT NOT NULL,
                    payload BLOB NOT NULL,
                    metadata_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations (id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS artifacts_conversation_idx
                    ON artifacts (conversation_id, id);

                PRAGMA user_version = 5;
                """
            )
        self.db_path.chmod(0o600)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create_user(self, username: str, password: str) -> User:
        cleaned_username = validate_username(username)
        validate_password(password)
        user_id = uuid4().hex
        created_at = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id, username, normalized_username, password_hash, created_at
                    )
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        cleaned_username,
                        normalize_username(cleaned_username),
                        PASSWORD_HASHER.hash(password),
                        created_at,
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UsernameAlreadyExistsError(
                "That username is already registered."
            ) from exc
        return User(id=user_id, username=cleaned_username, created_at=created_at)

    def authenticate_user(self, username: str, password: str) -> User | None:
        normalized_username = normalize_username(username)
        if not normalized_username or len(password) > MAX_PASSWORD_LENGTH:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, password_hash, created_at
                FROM users
                WHERE normalized_username = ?
                """,
                (normalized_username,),
            ).fetchone()
            if row is None:
                return None
            try:
                PASSWORD_HASHER.verify(row["password_hash"], password)
            except (InvalidHashError, VerificationError):
                return None
            if PASSWORD_HASHER.check_needs_rehash(row["password_hash"]):
                connection.execute(
                    "UPDATE users SET password_hash = ? WHERE id = ?",
                    (PASSWORD_HASHER.hash(password), row["id"]),
                )
        return User(
            id=row["id"],
            username=row["username"],
            created_at=row["created_at"],
        )

    def get_user(self, user_id: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, created_at
                FROM users
                WHERE id = ?
                """,
                (user_id,),
            ).fetchone()
        return None if row is None else User(**dict(row))

    @staticmethod
    def _auth_session_token_hash(token: str) -> str | None:
        if len(token) != AUTH_SESSION_TOKEN_LENGTH or not token.isascii():
            return None
        return sha256(token.encode("ascii")).hexdigest()

    def create_auth_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(AUTH_SESSION_TOKEN_BYTES)
        token_hash = self._auth_session_token_hash(token)
        if token_hash is None:  # pragma: no cover - token_urlsafe is ASCII.
            raise RuntimeError("Unable to create a valid authentication token.")
        created_at = datetime.now(UTC)
        expires_at = created_at + AUTH_SESSION_TTL
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?",
                (created_at.isoformat(),),
            )
            connection.execute(
                """
                INSERT INTO auth_sessions (
                    token_hash, user_id, created_at, expires_at
                )
                VALUES (?, ?, ?, ?)
                """,
                (
                    token_hash,
                    user_id,
                    created_at.isoformat(),
                    expires_at.isoformat(),
                ),
            )
        return token

    def get_user_by_auth_session(self, token: str) -> User | None:
        token_hash = self._auth_session_token_hash(token)
        if token_hash is None:
            return None
        now = datetime.now(UTC).isoformat()
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE expires_at <= ?",
                (now,),
            )
            row = connection.execute(
                """
                SELECT users.id, users.username, users.created_at
                FROM auth_sessions
                JOIN users ON users.id = auth_sessions.user_id
                WHERE auth_sessions.token_hash = ?
                    AND auth_sessions.expires_at > ?
                """,
                (token_hash, now),
            ).fetchone()
        return None if row is None else User(**dict(row))

    def revoke_auth_session(self, token: str) -> None:
        token_hash = self._auth_session_token_hash(token)
        if token_hash is None:
            return
        with self._connect() as connection:
            connection.execute(
                "DELETE FROM auth_sessions WHERE token_hash = ?",
                (token_hash,),
            )

    def create_conversation(self, user_id: str) -> Conversation:
        conversation_id = uuid4().hex
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations (
                    id, user_id, title, display_messages_json,
                    model_messages_json, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    user_id,
                    "New conversation",
                    "[]",
                    ModelMessagesTypeAdapter.dump_json([]),
                    now,
                    now,
                ),
            )
        return Conversation(
            id=conversation_id,
            user_id=user_id,
            title="New conversation",
            created_at=now,
            updated_at=now,
        )

    def list_conversations(self, user_id: str) -> list[ConversationSummary]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, title, updated_at
                FROM conversations
                WHERE user_id = ?
                  AND display_messages_json <> '[]'
                ORDER BY updated_at DESC
                """,
                (user_id,),
            ).fetchall()
        return [ConversationSummary(**dict(row)) for row in rows]

    def load_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> Conversation:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, title, display_messages_json,
                       model_messages_json, created_at, updated_at
                FROM conversations
                WHERE user_id = ? AND id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
        if row is None:
            raise ConversationNotFoundError(conversation_id)
        return self._conversation_from_row(row)

    def delete_conversation(
        self,
        user_id: str,
        conversation_id: str,
    ) -> None:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM conversations
                WHERE user_id = ? AND id = ?
                """,
                (user_id, conversation_id),
            )
            if cursor.rowcount != 1:
                raise ConversationNotFoundError(conversation_id)

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        display_messages = DISPLAY_MESSAGES_ADAPTER.validate_json(
            row["display_messages_json"]
        )
        model_messages = ModelMessagesTypeAdapter.validate_json(
            row["model_messages_json"]
        )
        return Conversation(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            display_messages=display_messages,
            model_messages=model_messages,
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    def save_conversation(
        self,
        user_id: str,
        conversation_id: str,
        display_messages: list[DisplayMessage],
        model_messages: list[ModelMessage],
    ) -> Conversation:
        now = self._now()
        serialized_display, serialized_model = self._serialize_conversation(
            display_messages,
            model_messages,
        )
        with self._connect() as connection:
            self._update_conversation(
                connection,
                user_id,
                conversation_id,
                display_messages,
                serialized_display,
                serialized_model,
                now,
            )
        return self.load_conversation(user_id, conversation_id)

    @staticmethod
    def _serialize_conversation(
        display_messages: list[DisplayMessage],
        model_messages: list[ModelMessage],
    ) -> tuple[str, bytes]:
        return (
            json.dumps(
                [message.model_dump(mode="json") for message in display_messages],
                ensure_ascii=False,
            ),
            ModelMessagesTypeAdapter.dump_json(model_messages),
        )

    @staticmethod
    def _update_conversation(
        connection: sqlite3.Connection,
        user_id: str,
        conversation_id: str,
        display_messages: list[DisplayMessage],
        serialized_display: str,
        serialized_model: bytes,
        now: str,
    ) -> None:
        cursor = connection.execute(
            """
            UPDATE conversations
            SET title = ?, display_messages_json = ?,
                model_messages_json = ?, updated_at = ?
            WHERE user_id = ? AND id = ?
            """,
            (
                conversation_title(display_messages),
                serialized_display,
                serialized_model,
                now,
                user_id,
                conversation_id,
            ),
        )
        if cursor.rowcount != 1:
            raise ConversationNotFoundError(conversation_id)

    def append_user(
        self,
        user_id: str,
        conversation_id: str,
        prompt: str,
    ) -> Conversation:
        """First save: append only the submitted user display message."""
        conversation = self.load_conversation(user_id, conversation_id)
        display_messages = [
            *conversation.display_messages,
            UserDisplayMessage(content=prompt),
        ]
        return self.save_conversation(
            user_id,
            conversation_id,
            display_messages,
            conversation.model_messages,
        )

    def commit_assistant(
        self,
        user_id: str,
        conversation_id: str,
        assistant: AssistantDisplayMessage,
        model_messages: list[ModelMessage],
        artifacts: Iterable[Artifact] = (),
    ) -> Conversation:
        """Second save: atomically persist a terminal turn and its artifacts."""
        conversation = self.load_conversation(user_id, conversation_id)
        display_messages = [*conversation.display_messages, assistant]
        now = self._now()
        serialized_display, serialized_model = self._serialize_conversation(
            display_messages,
            model_messages,
        )
        with self._connect() as connection:
            self._update_conversation(
                connection,
                user_id,
                conversation_id,
                display_messages,
                serialized_display,
                serialized_model,
                now,
            )
            for artifact in artifacts:
                self._insert_artifact(
                    connection,
                    conversation_id,
                    artifact,
                    now,
                )
        return self.load_conversation(user_id, conversation_id)

    def save_artifact(
        self,
        user_id: str,
        conversation_id: str,
        artifact: Artifact,
    ) -> Artifact:
        with self._connect() as connection:
            owner = connection.execute(
                """
                SELECT 1 FROM conversations
                WHERE user_id = ? AND id = ?
                """,
                (user_id, conversation_id),
            ).fetchone()
            if owner is None:
                raise ConversationNotFoundError(conversation_id)
            self._insert_artifact(
                connection,
                conversation_id,
                artifact,
                self._now(),
            )
        return artifact

    @staticmethod
    def _insert_artifact(
        connection: sqlite3.Connection,
        conversation_id: str,
        artifact: Artifact,
        created_at: str,
    ) -> None:
        content = artifact.require_content()
        if isinstance(content, TableContent):
            payload = content.arrow_bytes
            metadata = {
                "row_count": content.row_count,
                "truncated": content.truncated,
            }
        else:
            raise TypeError(
                f"SQLite storage does not support {artifact.kind!r} artifacts yet."
            )
        metadata_json = json.dumps(metadata)
        cursor = connection.execute(
            """
            INSERT INTO artifacts (
                id, conversation_id, kind, media_type,
                payload, metadata_json, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO NOTHING
            """,
            (
                artifact.id,
                conversation_id,
                artifact.kind,
                content.media_type,
                payload,
                metadata_json,
                created_at,
            ),
        )
        if cursor.rowcount == 1:
            return

        existing = connection.execute(
            """
            SELECT conversation_id, kind, media_type, payload, metadata_json
            FROM artifacts
            WHERE id = ?
            """,
            (artifact.id,),
        ).fetchone()
        if existing is None or (
            existing["conversation_id"] != conversation_id
            or existing["kind"] != artifact.kind
            or existing["media_type"] != content.media_type
            or bytes(existing["payload"]) != payload
            or json.loads(existing["metadata_json"]) != metadata
        ):
            raise sqlite3.IntegrityError(
                f"Artifact id {artifact.id!r} is already used by different content."
            )

    def load_artifact(
        self,
        user_id: str,
        conversation_id: str,
        artifact_id: str,
    ) -> Artifact:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.id, a.kind, a.media_type, a.payload, a.metadata_json
                FROM artifacts AS a
                JOIN conversations AS c ON c.id = a.conversation_id
                WHERE c.user_id = ?
                  AND a.conversation_id = ?
                  AND a.id = ?
                """,
                (user_id, conversation_id, artifact_id),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(artifact_id)
        if row["kind"] == "table" and row["media_type"] == ARROW_STREAM_MEDIA_TYPE:
            metadata = json.loads(row["metadata_json"])
            ref = ArtifactRef(
                artifact_id=row["id"],
                artifact_kind="table",
            )
            return Artifact(
                ref=ref,
                content=TableContent(
                    arrow_bytes=bytes(row["payload"]),
                    row_count=int(metadata["row_count"]),
                    truncated=bool(metadata["truncated"]),
                ),
            )
        raise TypeError(
            f"SQLite storage cannot load artifact kind {row['kind']!r} "
            f"with media type {row['media_type']!r}."
        )

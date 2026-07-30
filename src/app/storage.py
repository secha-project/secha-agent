from __future__ import annotations

import json
import os
import re
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter

from .dataframes import DataframeArtifact
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

                CREATE TABLE IF NOT EXISTS dataframe_queries (
                    id INTEGER PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sql TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations (id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS dataframe_queries_conversation_idx
                    ON dataframe_queries (conversation_id, id);

                CREATE TABLE IF NOT EXISTS dataframe_artifacts (
                    id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    arrow_bytes BLOB NOT NULL,
                    visible_row_count INTEGER NOT NULL
                        CHECK (visible_row_count >= 0),
                    truncated INTEGER NOT NULL CHECK (truncated IN (0, 1)),
                    created_at TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations (id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS artifacts_conversation_idx
                    ON dataframe_artifacts (conversation_id, id);

                PRAGMA user_version = 2;
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
        artifacts: Iterable[DataframeArtifact] = (),
    ) -> Conversation:
        """Second save: atomically persist a terminal assistant turn and artifacts."""
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

    def save_dataframe_query(self, conversation_id: str, sql: str) -> int:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO dataframe_queries (conversation_id, sql, created_at)
                SELECT id, ?, ?
                FROM conversations
                WHERE id = ?
                """,
                (sql, self._now(), conversation_id),
            )
            if cursor.rowcount != 1 or cursor.lastrowid is None:
                raise ConversationNotFoundError(conversation_id)
            return int(cursor.lastrowid)

    def load_dataframe_query(
        self,
        conversation_id: str,
        query_id: int,
    ) -> str | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT sql
                FROM dataframe_queries
                WHERE conversation_id = ? AND id = ?
                """,
                (conversation_id, query_id),
            ).fetchone()
        return None if row is None else str(row["sql"])

    def save_artifact(
        self,
        user_id: str,
        conversation_id: str,
        artifact: DataframeArtifact,
    ) -> DataframeArtifact:
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
        artifact: DataframeArtifact,
        created_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO dataframe_artifacts (
                id, conversation_id, arrow_bytes, visible_row_count,
                truncated, created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                artifact.id,
                conversation_id,
                artifact.arrow_bytes,
                artifact.visible_row_count,
                int(artifact.truncated),
                created_at,
            ),
        )

    def load_artifact(
        self,
        user_id: str,
        conversation_id: str,
        artifact_id: str,
    ) -> DataframeArtifact:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT a.id, a.arrow_bytes, a.visible_row_count, a.truncated
                FROM dataframe_artifacts AS a
                JOIN conversations AS c ON c.id = a.conversation_id
                WHERE c.user_id = ?
                  AND a.conversation_id = ?
                  AND a.id = ?
                """,
                (user_id, conversation_id, artifact_id),
            ).fetchone()
        if row is None:
            raise ArtifactNotFoundError(artifact_id)
        return DataframeArtifact(
            id=row["id"],
            arrow_bytes=bytes(row["arrow_bytes"]),
            visible_row_count=row["visible_row_count"],
            truncated=bool(row["truncated"]),
        )

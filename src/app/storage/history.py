from __future__ import annotations

import json
import os
import re
import secrets
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from functools import cache
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from argon2 import PasswordHasher
from argon2.exceptions import InvalidHashError, VerificationError
from pydantic_ai import ModelMessage, ModelMessagesTypeAdapter

from src.agent.outputs.artifacts import (
    Artifact,
    ChartArtifact,
    TableArtifact,
)

from ..models import (
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


def public_signup_enabled() -> bool:
    return os.environ.get("SECHA_ALLOW_SIGNUP", "false").strip().casefold() == "true"


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


class History:
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
                Path(__file__).with_name("schema.sql").read_text(encoding="utf-8")
            )
            connection.execute("BEGIN IMMEDIATE")
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(users)")
            }
            for column in ("is_admin", "must_change_password"):
                if column not in columns:
                    connection.execute(
                        f"ALTER TABLE users ADD COLUMN {column} INTEGER NOT NULL DEFAULT 0"
                    )
        self.db_path.chmod(0o600)

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def register_user(self, username: str, password: str) -> User:
        if not public_signup_enabled():
            raise PermissionError(
                "Public signup is disabled. Contact your administrator."
            )
        return self.create_user(username, password)

    def require_admin(self, token: str) -> User:
        user = self.get_user_by_auth_session(token)
        if user is None or not user.is_admin or user.must_change_password:
            raise PermissionError("Administrator access is required.")
        return user

    def create_managed_user(self, token: str, username: str, password: str) -> User:
        self.require_admin(token)
        return self.create_user(username, password, must_change_password=True)

    def list_users(self, token: str) -> list[User]:
        self.require_admin(token)
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT id, username, created_at, is_admin, must_change_password "
                "FROM users ORDER BY normalized_username"
            ).fetchall()
        return [User(**dict(row)) for row in rows]

    def change_password(
        self, token: str, current_password: str, new_password: str
    ) -> User:
        user = self.get_user_by_auth_session(token)
        if (
            user is None
            or self.authenticate_user(user.username, current_password) is None
        ):
            raise PermissionError("Your current password or session is invalid.")
        validate_password(new_password)
        if current_password == new_password:
            raise ValueError(
                "Choose a different password from your temporary password."
            )
        with self._connect() as connection:
            changed = connection.execute(
                "UPDATE users SET password_hash = ?, must_change_password = 0 "
                "WHERE id = ? AND EXISTS (SELECT 1 FROM auth_sessions "
                "WHERE user_id = users.id AND token_hash = ? AND expires_at > ?)",
                (
                    PASSWORD_HASHER.hash(new_password),
                    user.id,
                    self._auth_session_token_hash(token),
                    self._now(),
                ),
            )
            if changed.rowcount != 1:
                raise PermissionError("Your session has expired. Please log in again.")
            connection.execute(
                "DELETE FROM auth_sessions WHERE user_id = ?", (user.id,)
            )
        return user.model_copy(update={"must_change_password": False})

    def create_user(
        self, username: str, password: str, *, must_change_password: bool = False
    ) -> User:
        """Trusted local provisioning; UI callers must use the guarded methods above."""
        cleaned_username = validate_username(username)
        validate_password(password)
        user_id = uuid4().hex
        created_at = self._now()
        try:
            with self._connect() as connection:
                connection.execute(
                    """
                    INSERT INTO users (
                        id, username, normalized_username, password_hash, created_at, must_change_password
                    )
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        user_id,
                        cleaned_username,
                        normalize_username(cleaned_username),
                        PASSWORD_HASHER.hash(password),
                        created_at,
                        int(must_change_password),
                    ),
                )
        except sqlite3.IntegrityError as exc:
            raise UsernameAlreadyExistsError(
                "That username is already registered."
            ) from exc
        return User(
            id=user_id,
            username=cleaned_username,
            created_at=created_at,
            must_change_password=must_change_password,
        )

    def authenticate_user(self, username: str, password: str) -> User | None:
        normalized_username = normalize_username(username)
        if not normalized_username or len(password) > MAX_PASSWORD_LENGTH:
            return None
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, password_hash, created_at, is_admin, must_change_password
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
            is_admin=bool(row["is_admin"]),
            must_change_password=bool(row["must_change_password"]),
        )

    def get_user(self, user_id: str) -> User | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, username, created_at, is_admin, must_change_password
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
                SELECT users.id, users.username, users.created_at,
                    users.is_admin, users.must_change_password
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

    @staticmethod
    def _insert_artifact(
        connection: sqlite3.Connection,
        conversation_id: str,
        artifact: Artifact,
        created_at: str,
    ) -> None:
        if isinstance(artifact, TableArtifact):
            kind = "table"
            payload = artifact.arrow_bytes
            metadata = {
                "truncated": artifact.truncated,
            }
            media_type = TableArtifact.MEDIA_TYPE
        elif isinstance(artifact, ChartArtifact):
            kind = "chart"
            payload = artifact.spec_json
            metadata = {}
            media_type = ChartArtifact.MEDIA_TYPE
        else:
            raise TypeError(
                f"SQLite storage does not support {type(artifact).__name__}."
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
                kind,
                media_type,
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
            or existing["kind"] != kind
            or existing["media_type"] != media_type
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
        if row["kind"] == "table" and row["media_type"] == TableArtifact.MEDIA_TYPE:
            metadata = json.loads(row["metadata_json"])
            return TableArtifact(
                id=row["id"],
                arrow_bytes=bytes(row["payload"]),
                truncated=bool(metadata["truncated"]),
            )
        if row["kind"] == "chart" and row["media_type"] == ChartArtifact.MEDIA_TYPE:
            spec = json.loads(row["payload"])
            if not isinstance(spec, dict):
                raise TypeError("Saved Vega-Lite content must be a JSON object.")
            return ChartArtifact(
                id=row["id"],
                spec=spec,
            )
        raise TypeError(
            f"SQLite storage cannot load artifact kind {row['kind']!r} "
            f"with media type {row['media_type']!r}."
        )


@cache
def load_history() -> History:
    """Return the process-wide history shared by UI and HTTP routes."""
    history = History()
    history.initialize()
    return history

from pathlib import Path

from starlette.applications import Starlette
from starlette.testclient import TestClient

from src.app.routes.auth import (
    AUTH_COOKIE_NAME,
    AUTH_REQUEST_HEADER,
    create_auth_routes,
)
from src.app.storage.history import AUTH_SESSION_TTL_SECONDS, History


def test_auth_cookie_is_http_only_and_revoked_on_logout(tmp_path: Path) -> None:
    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    user = history.create_user("Alice", "correct-horse")
    token = history.create_auth_session(user.id)
    app = Starlette(routes=create_auth_routes(history))
    headers = {AUTH_REQUEST_HEADER: "1"}

    with TestClient(app) as client:
        response = client.post("/auth/session", content=token, headers=headers)

        assert response.status_code == 204
        assert response.headers["cache-control"] == "no-store"
        cookie = response.headers["set-cookie"]
        assert f"{AUTH_COOKIE_NAME}={token}" in cookie
        assert f"Max-Age={AUTH_SESSION_TTL_SECONDS}" in cookie
        assert "HttpOnly" in cookie
        assert "SameSite=lax" in cookie
        assert "Secure" not in cookie

        response = client.post("/auth/logout", headers=headers)

        assert response.status_code == 204
        assert history.get_user_by_auth_session(token) is None
        assert client.cookies.get(AUTH_COOKIE_NAME) is None


def test_auth_cookie_rejects_unknown_token(tmp_path: Path) -> None:
    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    app = Starlette(routes=create_auth_routes(history))

    with TestClient(app) as client:
        response = client.post(
            "/auth/session",
            content="x" * 43,
            headers={AUTH_REQUEST_HEADER: "1"},
        )

    assert response.status_code == 401
    assert AUTH_COOKIE_NAME not in response.headers.get("set-cookie", "")


def test_auth_cookie_is_secure_over_https(tmp_path: Path) -> None:
    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    user = history.create_user("Alice", "correct-horse")
    token = history.create_auth_session(user.id)
    app = Starlette(routes=create_auth_routes(history))

    with TestClient(app, base_url="https://testserver") as client:
        response = client.post(
            "/auth/session",
            content=token,
            headers={AUTH_REQUEST_HEADER: "1"},
        )

    assert response.status_code == 204
    assert "Secure" in response.headers["set-cookie"]


def test_auth_cookie_routes_reject_cross_site_style_requests(
    tmp_path: Path,
) -> None:
    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    user = history.create_user("Alice", "correct-horse")
    token = history.create_auth_session(user.id)
    app = Starlette(routes=create_auth_routes(history))

    with TestClient(app) as client:
        create_response = client.post("/auth/session", content=token)
        logout_response = client.post("/auth/logout")

    assert create_response.status_code == 403
    assert logout_response.status_code == 403
    assert history.get_user_by_auth_session(token) == user


def test_managed_accounts_and_password_change(tmp_path, monkeypatch):
    import pytest

    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    monkeypatch.delenv("SECHA_ALLOW_SIGNUP", raising=False)
    with pytest.raises(PermissionError):
        history.register_user("Public", "correct-horse")
    monkeypatch.setenv("SECHA_ALLOW_SIGNUP", "true")
    admin = history.register_user("Owner", "correct-horse")
    token = history.create_auth_session(admin.id)
    with pytest.raises(PermissionError):
        history.create_managed_user(token, "Blocked", "correct-horse")
    with pytest.raises(PermissionError):
        history.list_users(token)
    with history._connect() as connection:
        connection.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (admin.id,))
    monkeypatch.setenv("SECHA_ALLOW_SIGNUP", "false")
    user = history.create_managed_user(token, "Alice", "temporary-password")
    assert user.must_change_password and not user.is_admin
    assert history.authenticate_user("Alice", "temporary-password") == user
    assert len(history.list_users(token)) == 2
    user_token = history.create_auth_session(user.id)
    another_token = history.create_auth_session(user.id)
    with pytest.raises(PermissionError):
        history.change_password(user_token, "wrong-password", "lasting-password")
    with pytest.raises(ValueError):
        history.change_password(user_token, "temporary-password", "short")
    with pytest.raises(ValueError):
        history.change_password(user_token, "temporary-password", "temporary-password")
    updated = history.change_password(
        user_token, "temporary-password", "lasting-password"
    )
    assert not updated.must_change_password
    assert history.authenticate_user("Alice", "temporary-password") is None
    assert history.authenticate_user("Alice", "lasting-password") == updated
    assert history.get_user_by_auth_session(user_token) is None
    assert history.get_user_by_auth_session(another_token) is None
    history.revoke_auth_session(token)
    with pytest.raises(PermissionError):
        history.create_managed_user(token, "Blocked", "correct-horse")


def test_existing_database_migrates_without_changing_accounts(tmp_path):
    import sqlite3

    from src.app.storage.history import PASSWORD_HASHER

    path = tmp_path / "legacy.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE users (id TEXT PRIMARY KEY, username TEXT NOT NULL, "
            "normalized_username TEXT UNIQUE NOT NULL, password_hash TEXT NOT NULL, "
            "created_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO users VALUES (?, ?, ?, ?, ?)",
            (
                "old-user",
                "Alice",
                "alice",
                PASSWORD_HASHER.hash("correct-horse"),
                "2026-01-01",
            ),
        )
    history = History(path)
    history.initialize()
    history.initialize()
    user = history.authenticate_user("Alice", "correct-horse")
    assert user.id == "old-user"
    assert not user.is_admin and not user.must_change_password


def test_signup_hidden_by_default(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    monkeypatch.setenv("SECHA_CHAT_HISTORY_DB", str(tmp_path / "ui.sqlite3"))
    monkeypatch.delenv("SECHA_ALLOW_SIGNUP", raising=False)
    app = AppTest.from_string("""
from src.app.storage.history import History
from src.app.views.authentication import render_authentication
history = History()
history.initialize()
render_authentication(history)
""").run()
    assert not app.exception
    assert [button.label for button in app.button] == ["Log in"]
    assert not app.get("segmented_control")


def test_cached_identity_does_not_bypass_revoked_session(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from src.app import state

    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    user = history.create_user("Alice", "correct-horse")
    token = history.create_auth_session(user.id)
    session_state = {state.AUTHENTICATED_USER_KEY: user.id}
    monkeypatch.setattr(
        state,
        "st",
        SimpleNamespace(
            session_state=session_state,
            query_params={},
            context=SimpleNamespace(cookies={AUTH_COOKIE_NAME: token}),
        ),
    )
    assert state.current_user(history) == user
    history.revoke_auth_session(token)
    assert state.current_user(history) is None
    assert state.AUTHENTICATED_USER_KEY not in session_state


def test_temporary_password_blocks_app_before_resources_load(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest

    import src.app.storage.history as storage
    from src.app import resources, state

    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    user = history.create_user("Alice", "temporary-password", must_change_password=True)
    monkeypatch.setattr(storage, "load_history", lambda: history)
    monkeypatch.setattr(state, "current_user", lambda _: user)
    monkeypatch.setattr(
        resources,
        "load_resources",
        lambda: (_ for _ in ()).throw(
            AssertionError("App resources must not load before changing the password")
        ),
    )
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/app/application.py"
    ).run()
    assert not app.exception
    assert app.title[0].value == "Choose your password"


def test_admin_command_promotes_existing_account(tmp_path, monkeypatch):
    from src.app import admin

    history = History(tmp_path / "chat.sqlite3")
    history.initialize()
    user = history.create_user("Alice", "correct-horse")
    monkeypatch.setattr("sys.argv", ["admin", "alice", "--db", str(history.db_path)])
    admin.main()
    assert history.get_user(user.id).is_admin
    assert history.authenticate_user("Alice", "correct-horse").is_admin


def test_users_navigation_returns_to_new_and_existing_chats(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from streamlit.testing.v1 import AppTest

    import src.app.storage.history as storage
    from src.app import resources, state

    history = History(tmp_path / "navigation.sqlite3")
    history.initialize()
    user = history.create_user("Owner", "correct-horse")
    with history._connect() as connection:
        connection.execute("UPDATE users SET is_admin = 1 WHERE id = ?", (user.id,))
    user = history.get_user(user.id)
    token = history.create_auth_session(user.id)
    conversation = history.create_conversation(user.id)
    history.append_user(user.id, conversation.id, "Saved question")
    monkeypatch.setattr(storage, "load_history", lambda: history)
    monkeypatch.setattr(state, "current_user", lambda _: user)
    monkeypatch.setattr(state, "auth_cookie_token", lambda: token)
    # Patch the imported alias too when a prior test has already loaded the view.
    import src.app.views.users as users_view

    monkeypatch.setattr(users_view, "auth_cookie_token", lambda: token)
    monkeypatch.setattr(
        resources, "load_resources", lambda: SimpleNamespace(history=history)
    )
    app = AppTest.from_file(
        Path(__file__).resolve().parents[1] / "src/app/application.py"
    ).run()

    def click_sidebar(label):
        next(
            button for button in app.sidebar.button if button.label == label
        ).click().run()
        assert not app.exception

    click_sidebar("Users")
    assert app.title[0].value == "Users"
    click_sidebar("New chat")
    assert len(app.chat_input) == 1
    assert app.session_state[state.APP_VIEW_KEY] == "chat"
    click_sidebar("Users")
    app.button(key=f"conversation-{conversation.id}").click().run()
    assert not app.exception
    assert len(app.chat_input) == 1
    assert app.session_state.current_conversation_id == conversation.id
    assert app.query_params["chat"] == [conversation.id]
    click_sidebar("Users")
    assert app.title[0].value == "Users"

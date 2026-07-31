from pathlib import Path

from starlette.applications import Starlette
from starlette.testclient import TestClient

from src.app.auth import AUTH_COOKIE_NAME, AUTH_REQUEST_HEADER, create_auth_routes
from src.app.storage import AUTH_SESSION_TTL_SECONDS, ChatRepository


def test_auth_cookie_is_http_only_and_revoked_on_logout(tmp_path: Path) -> None:
    repository = ChatRepository(tmp_path / "chat.sqlite3")
    repository.initialize()
    user = repository.create_user("Alice", "correct-horse")
    token = repository.create_auth_session(user.id)
    app = Starlette(routes=create_auth_routes(repository))
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
        assert repository.get_user_by_auth_session(token) is None
        assert client.cookies.get(AUTH_COOKIE_NAME) is None


def test_auth_cookie_rejects_unknown_token(tmp_path: Path) -> None:
    repository = ChatRepository(tmp_path / "chat.sqlite3")
    repository.initialize()
    app = Starlette(routes=create_auth_routes(repository))

    with TestClient(app) as client:
        response = client.post(
            "/auth/session",
            content="x" * 43,
            headers={AUTH_REQUEST_HEADER: "1"},
        )

    assert response.status_code == 401
    assert AUTH_COOKIE_NAME not in response.headers.get("set-cookie", "")


def test_auth_cookie_is_secure_over_https(tmp_path: Path) -> None:
    repository = ChatRepository(tmp_path / "chat.sqlite3")
    repository.initialize()
    user = repository.create_user("Alice", "correct-horse")
    token = repository.create_auth_session(user.id)
    app = Starlette(routes=create_auth_routes(repository))

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
    repository = ChatRepository(tmp_path / "chat.sqlite3")
    repository.initialize()
    user = repository.create_user("Alice", "correct-horse")
    token = repository.create_auth_session(user.id)
    app = Starlette(routes=create_auth_routes(repository))

    with TestClient(app) as client:
        create_response = client.post("/auth/session", content=token)
        logout_response = client.post("/auth/logout")

    assert create_response.status_code == 403
    assert logout_response.status_code == 403
    assert repository.get_user_by_auth_session(token) == user

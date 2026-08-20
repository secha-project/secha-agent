from __future__ import annotations

import os
from collections.abc import Sequence

from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Route

from src.app.storage.history import AUTH_SESSION_TTL_SECONDS, History

AUTH_COOKIE_NAME = "secha_session"
AUTH_SESSION_ENDPOINT = "/auth/session"
AUTH_LOGOUT_ENDPOINT = "/auth/logout"
AUTH_REQUEST_HEADER = "X-SECHA-Auth"
MAX_AUTH_REQUEST_BYTES = 128


def _use_secure_cookie(request: Request) -> bool:
    configured = os.environ.get("SECHA_AUTH_SECURE_COOKIE", "").strip().casefold()
    return request.url.scheme == "https" or configured in {"1", "true", "yes"}


def create_auth_routes(history: History) -> Sequence[BaseRoute]:
    async def create_session_cookie(request: Request) -> Response:
        if request.headers.get(AUTH_REQUEST_HEADER) != "1":
            return Response(status_code=403)
        raw_token = await request.body()
        if len(raw_token) > MAX_AUTH_REQUEST_BYTES:
            return Response(status_code=400)
        try:
            token = raw_token.decode("ascii")
        except UnicodeDecodeError:
            return Response(status_code=400)
        if history.get_user_by_auth_session(token) is None:
            return Response(status_code=401)

        response = Response(status_code=204)
        response.set_cookie(
            AUTH_COOKIE_NAME,
            token,
            max_age=AUTH_SESSION_TTL_SECONDS,
            path="/",
            secure=_use_secure_cookie(request),
            httponly=True,
            samesite="lax",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    async def clear_session_cookie(request: Request) -> Response:
        if request.headers.get(AUTH_REQUEST_HEADER) != "1":
            return Response(status_code=403)
        token = request.cookies.get(AUTH_COOKIE_NAME)
        if token:
            history.revoke_auth_session(token)
        response = Response(status_code=204)
        response.delete_cookie(
            AUTH_COOKIE_NAME,
            path="/",
            secure=_use_secure_cookie(request),
            httponly=True,
            samesite="lax",
        )
        response.headers["Cache-Control"] = "no-store"
        return response

    return (
        Route(AUTH_SESSION_ENDPOINT, create_session_cookie, methods=["POST"]),
        Route(AUTH_LOGOUT_ENDPOINT, clear_session_cookie, methods=["POST"]),
    )

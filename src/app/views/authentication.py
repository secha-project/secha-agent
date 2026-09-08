from __future__ import annotations

import json

import streamlit as st

from ..models import User
from ..routes.auth import (
    AUTH_LOGOUT_ENDPOINT,
    AUTH_REQUEST_HEADER,
    AUTH_SESSION_ENDPOINT,
)
from ..state import (
    auth_cookie_token,
    clear_authenticated_session,
    set_authenticated_user,
)
from ..storage.history import (
    History,
    InvalidUsernameError,
    UsernameAlreadyExistsError,
    WeakPasswordError,
    public_signup_enabled,
)


def render_auth_request(endpoint: str, token: str | None = None) -> None:
    """Use a same-origin request to update an HTTP-only authentication cookie."""
    body = "null" if token is None else json.dumps(token)
    st.html(
        f"""
        <p id="auth-transition">Updating your session…</p>
        <script>
        (async () => {{
            const options = {{
                method: "POST",
                credentials: "same-origin",
                headers: {{{json.dumps(AUTH_REQUEST_HEADER)}: "1"}},
            }};
            const body = {body};
            if (body !== null) {{
                options.headers["Content-Type"] = "text/plain";
                options.body = body;
            }}
            const response = await fetch({json.dumps(endpoint)}, options);
            if (!response.ok) {{
                document.getElementById("auth-transition").textContent =
                    "Unable to update the login cookie. Start the app via " +
                    "streamlit_app.py and try again.";
                return;
            }}
            window.location.reload();
        }})();
        </script>
        """,
        unsafe_allow_javascript=True,
    )
    st.stop()


def complete_login(history: History, user: User) -> None:
    set_authenticated_user(user)
    token = history.create_auth_session(user.id)
    render_auth_request(AUTH_SESSION_ENDPOINT, token)


def render_authentication(history: History) -> None:
    st.title("SECHA")
    st.caption("Sign in to continue to your analytics assistant.")
    mode = "Log in"
    if public_signup_enabled():
        mode = st.segmented_control(
            "Account action",
            ["Log in", "Sign up"],
            default="Log in",
            required=True,
            key="auth-mode",
            label_visibility="collapsed",
            width="stretch",
        )

    with st.container(border=True):
        if mode == "Sign up":
            st.subheader("Create an account")
            with st.form("signup-form", border=False):
                username = st.text_input(
                    "Username",
                    max_chars=32,
                    autocomplete="username",
                    icon=":material/person:",
                )
                password = st.text_input(
                    "Password",
                    type="password",
                    max_chars=128,
                    autocomplete="new-password",
                    help="Use at least 8 characters.",
                    icon=":material/lock:",
                )
                confirmation = st.text_input(
                    "Confirm password",
                    type="password",
                    max_chars=128,
                    autocomplete="new-password",
                    icon=":material/lock:",
                )
                submitted = st.form_submit_button(
                    "Create account",
                    type="primary",
                    icon=":material/person_add:",
                    width="stretch",
                )
            if not submitted:
                return
            if password != confirmation:
                st.error("Passwords do not match.")
                return
            try:
                user = history.register_user(username, password)
            except (
                PermissionError,
                InvalidUsernameError,
                WeakPasswordError,
                UsernameAlreadyExistsError,
            ) as exc:
                st.error(str(exc))
                return
            complete_login(history, user)

        st.subheader("Log in")
        with st.form("login-form", border=False):
            username = st.text_input(
                "Username",
                max_chars=32,
                autocomplete="username",
                icon=":material/person:",
            )
            password = st.text_input(
                "Password",
                type="password",
                max_chars=128,
                autocomplete="current-password",
                icon=":material/lock:",
            )
            submitted = st.form_submit_button(
                "Log in",
                type="primary",
                icon=":material/login:",
                width="stretch",
            )
        if not submitted:
            return
        user = history.authenticate_user(username, password)
        if user is None:
            st.error("Invalid username or password.")
            return
        complete_login(history, user)


def render_password_change(history: History, user: User) -> None:
    st.title("Choose your password")
    st.caption(
        f"Welcome, {user.username}. Replace your temporary password to continue."
    )
    with st.form("change-password", clear_on_submit=True):
        current = st.text_input("Temporary password", type="password", max_chars=128)
        password = st.text_input(
            "New password",
            type="password",
            max_chars=128,
            help="Use at least 8 characters.",
        )
        confirmation = st.text_input(
            "Confirm new password", type="password", max_chars=128
        )
        submitted = st.form_submit_button("Save password", type="primary")
    if submitted:
        if password != confirmation:
            st.error("Passwords do not match.")
        else:
            try:
                updated = history.change_password(
                    auth_cookie_token() or "", current, password
                )
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
            else:
                complete_login(history, updated)
    if st.button("Sign out"):
        history.revoke_auth_session(auth_cookie_token() or "")
        clear_authenticated_session()
        render_auth_request(AUTH_LOGOUT_ENDPOINT)

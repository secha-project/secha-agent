from __future__ import annotations

import streamlit as st

from ..models import User
from ..resources import AppResources
from ..state import auth_cookie_token
from .sidebar import render_sidebar


def render_users(resources: AppResources, user: User) -> None:
    history = resources.history
    render_sidebar(resources, user)
    token = auth_cookie_token() or ""
    try:
        history.require_admin(token)
    except PermissionError as exc:
        st.error(str(exc))
        return

    st.space("medium")
    st.title("Users")
    st.caption(
        "Create an account and share its credentials privately with the recipient."
    )
    with st.form("create-managed-user", clear_on_submit=True):
        st.subheader("Create an account")
        username = st.text_input("Username", max_chars=32)
        password = st.text_input(
            "Temporary password",
            type="password",
            max_chars=128,
            help="Use a unique password of at least 8 characters. The user must replace it at first login.",
        )
        confirmation = st.text_input(
            "Confirm temporary password", type="password", max_chars=128
        )
        submitted = st.form_submit_button("Create account", type="primary")
    if submitted:
        if password != confirmation:
            st.error("Passwords do not match.")
        else:
            try:
                user = history.create_managed_user(token, username, password)
            except (ValueError, PermissionError) as exc:
                st.error(str(exc))
            else:
                st.success(
                    f"Account created for {user.username}. Password change required at first login."
                )

    try:
        users = history.list_users(token)
    except PermissionError as exc:
        st.error(str(exc))
        return
    st.subheader("Existing accounts")
    st.dataframe(
        [
            {
                "Username": user.username,
                "Admin": user.is_admin,
                "Password change required": user.must_change_password,
                "Created": user.created_at,
            }
            for user in users
        ],
        hide_index=True,
    )

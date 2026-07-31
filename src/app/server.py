from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app.auth import create_auth_routes
from src.app.storage import ChatRepository

repository = ChatRepository()
repository.initialize()

app = st.App(
    Path(__file__).with_name("main.py"),
    routes=create_auth_routes(repository),
)

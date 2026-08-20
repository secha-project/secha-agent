from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app.routes.auth import create_auth_routes
from src.app.storage.history import load_history

history = load_history()

app = st.App(
    Path(__file__).with_name("application.py"),
    routes=create_auth_routes(history),
)

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

import streamlit as st

from src.app.resources import shutdown_resources
from src.app.routes.auth import create_auth_routes
from src.app.storage.history import load_history

history = load_history()


@asynccontextmanager
async def lifespan(_app):
    try:
        yield
    finally:
        shutdown_resources()


app = st.App(
    Path(__file__).parent / "src/app/application.py",
    routes=create_auth_routes(history),
    lifespan=lifespan,
)

from __future__ import annotations

import streamlit as st

from src.app.resources import load_resources
from src.app.state import current_user
from src.app.storage.history import load_history
from src.app.styles import apply_styles
from src.app.views.authentication import render_authentication
from src.app.views.chat import render_chat

st.set_page_config(
    page_title="SECHA",
    page_icon=":material/battery_charging_full:",
)
apply_styles()

history = load_history()
user = current_user(history)
if user is None:
    render_authentication(history)
else:
    render_chat(load_resources(), user)

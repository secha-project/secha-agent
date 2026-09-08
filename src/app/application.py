from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.app.resources import load_resources
from src.app.state import APP_VIEW_KEY, current_user
from src.app.storage.history import load_history
from src.app.views.authentication import render_authentication, render_password_change
from src.app.views.chat import render_chat
from src.app.views.metric_views import render_metric_views
from src.app.views.users import render_users

st.set_page_config(
    page_title="SECHA",
    page_icon=":material/battery_charging_full:",
)
st.html(Path(__file__).with_name("styles.css"))

history = load_history()
user = current_user(history)
if user is None:
    render_authentication(history)
elif user.must_change_password:
    render_password_change(history, user)
else:
    resources = load_resources()
    if st.session_state.get(APP_VIEW_KEY) == "users":
        render_users(resources, user)
    elif st.session_state.get(APP_VIEW_KEY) == "metric_views":
        render_metric_views(resources, user)
    else:
        render_chat(resources, user)

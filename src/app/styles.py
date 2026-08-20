from __future__ import annotations

from pathlib import Path

import streamlit as st

STYLES_PATH = Path(__file__).with_name("styles.css")


@st.cache_data
def load_styles() -> str:
    return STYLES_PATH.read_text(encoding="utf-8")


def apply_styles() -> None:
    st.markdown(f"<style>{load_styles()}</style>", unsafe_allow_html=True)

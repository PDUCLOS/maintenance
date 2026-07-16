"""Streamlit UI — thin shell that wires the sidebar + 4 tabs together.

Run with:
    make ui    # http://localhost:8501

Each tab lives in its own module under `src/ui/tabs/`:
  - chat       — the core RAG chat (intent picker + free-form input)
  - inventory  — data/raw/pdf listing
  - ragas      — RAGAS evaluation snapshots
  - index      — ChromaDB stats

The shared sidebar (system status + "About this tool") lives in
`src/ui/sidebar.py`. The HTTP client (httpx wrapper around the
FastAPI backend on :8000) lives in `src/ui/api_client.py`.

This file is intentionally < 40 lines: its only job is to glue
the pieces together and call set_page_config() once.
"""

from __future__ import annotations

from pathlib import Path

import streamlit as st

from src.ui.sidebar import render as render_sidebar
from src.ui.tabs import chat, index, inventory, ragas

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


st.set_page_config(
    page_title="Industrial Knowledge Copilot",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_sidebar()

tab_chat, tab_inventory, tab_ragas, tab_index = st.tabs(
    ["💬 Chat", "📦 Inventaire", "📊 RAGAS", "🗂️ Index"]
)
with tab_chat:
    chat.render()
with tab_inventory:
    inventory.render()
with tab_ragas:
    ragas.render(REPORTS_DIR)
with tab_index:
    index.render()

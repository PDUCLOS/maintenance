"""Streamlit chat UI.

Run with:
    make ui    # http://localhost:8501

The UI calls the FastAPI backend on :8000. If the API is down, it shows
a clear error in the sidebar instead of pretending the request succeeded.
"""

from __future__ import annotations

import time

import requests
import streamlit as st

from src.config import settings

API_BASE = f"http://localhost:{settings.api_port}"


def _api_get(path: str, timeout: float = 5.0) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API call failed: {e}")
        return None


def _api_post(path: str, payload: dict, timeout: float = 60.0) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"API call failed: {e}")
        return None


# --- Page config ------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Knowledge Copilot",
    page_icon="🛠️",
    layout="wide",
)

# --- Sidebar ----------------------------------------------------------------
with st.sidebar:
    st.title("🛠️ IKC")
    st.caption("Local RAG copilot for industrial maintenance.")

    health = _api_get("/health")
    if health:
        st.write("**Status**")
        st.json({
            "status": health["status"],
            "chroma": health["chroma_reachable"],
            "mlx_ready": health["mlx_ready"],
            "chunks_indexed": health["collection_count"],
            "hardware": health["hardware"],
        })
    else:
        st.warning("API unreachable on :8000. Start it with `make api`.")

    st.divider()
    top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=20, value=5)
    use_agent = st.toggle("Use agent (with Python tool calling)", value=False,
                          help="Allows the copilot to run pandas queries on CMAPSS data.")
    st.divider()
    st.caption(f"Streamlit on :{settings.ui_port} · API on :{settings.api_port}")

# --- Main chat --------------------------------------------------------------
st.title("Industrial Knowledge Copilot")
st.caption("Pose une question sur la maintenance industrielle (NASA CMAPSS) ou sur les PDF techniques.")

# Chat history (Streamlit session_state)
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(f"Sources ({len(msg['sources'])})", expanded=False):
                for s in msg["sources"]:
                    st.markdown(
                        f"**`{s['id']}`** · `{s['source']}` · score={s['score']:.3f}\n\n"
                        f"> {s['text'][:400]}{'…' if len(s['text']) > 400 else ''}"
                    )

# Input
if prompt := st.chat_input("Ta question…"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        t0 = time.perf_counter()
        result = _api_post("/query", {"question": prompt, "top_k": top_k, "stream": False})
        elapsed = (time.perf_counter() - t0) * 1000
        if result is None:
            answer = "❌ L'API n'a pas répondu."
            sources = []
        else:
            answer = result.get("answer", "(pas de réponse)")
            sources = result.get("sources", [])
        st.markdown(answer)
        st.caption(f"⏱️ {elapsed:.0f} ms · {len(sources)} sources")
        if sources:
            with st.expander(f"Sources ({len(sources)})", expanded=False):
                for s in sources:
                    st.markdown(
                        f"**`{s['id']}`** · `{s['source']}` · score={s['score']:.3f}\n\n"
                        f"> {s['text'][:400]}{'…' if len(s['text']) > 400 else ''}"
                    )
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
        })

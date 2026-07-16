"""Index tab — ChromaDB stats: chunk count, source distribution, sample.

Reads /health and /index/stats from the API (no direct chromadb
client in this process — a second client from a non-main thread
inside Streamlit's rerun model segfaults the interpreter, see the
comment block below).

We display:
  - chunk count, chroma reachable flag, hardware (3 metrics),
  - source distribution as a bar chart (capped at 10k chunks for
    perf, see src/api/routes/index.py),
  - 5 sample chunks with metadata + text preview.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.ui.api_client import api_get


def _render_overview(health: dict) -> None:
    """Three columns: chunks, chroma reachable, hardware."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Chunks totaux", health.get("collection_count", 0))
    with col2:
        st.metric("Chroma joignable", "✅" if health.get("chroma_reachable") else "❌")
    with col3:
        st.metric("Hardware", health.get("hardware", "?"))


def _render_source_distribution(stats: dict) -> None:
    """Bar chart of chunk counts per source (PDF filename)."""
    source_counts = stats.get("source_counts", {})
    if not source_counts:
        st.info("La collection est vide.")
        return
    df_sources = pd.DataFrame(
        [{"source": k, "count": v} for k, v in source_counts.items()]
    ).sort_values("count", ascending=False)
    st.bar_chart(df_sources.set_index("source"))
    st.caption(
        f"Distribution sur {sum(source_counts.values())} chunks échantillonnés "
        f"(plafond 10k pour la perf)."
    )


def _render_sample_chunks(stats: dict) -> None:
    """5 sample chunks with metadata + text preview."""
    sample = stats.get("sample_chunks", [])
    if not sample:
        return
    for chunk in sample:
        with st.expander(f"`{chunk['id']}`", expanded=False):
            st.caption(f"Source : `{chunk['source']}` · Métadonnées : `{chunk['metadata']}`")
            text = chunk["text"]
            st.text(text[:600] + ("…" if len(text) > 600 else ""))


def render() -> None:
    """Render the Index tab content (called inside `with tab_index:`)."""
    st.title("🗂️ Index ChromaDB")
    st.caption(
        "État de l'index vectoriel en temps réel. Si la collection est vide, lance "
        "`make ingest` après avoir populé `data/raw/`."
    )

    health = api_get("/health")
    if not health:
        st.warning("API injoignable sur :8000. Lance `make api`.")
        return

    _render_overview(health)

    # Source distribution + sample chunks — via the API's /index/stats,
    # not a direct chromadb.HttpClient() from this process. A second
    # client instantiated from Streamlit's own thread reliably
    # segfaults the interpreter (chromadb's posthog telemetry call is
    # broken on this version and something about triggering it from a
    # non-main thread inside Streamlit's script-rerun model corrupts
    # the process — same family of issue as MLX's thread-bound Metal
    # stream elsewhere in this codebase). The API process already owns
    # one healthy chromadb client; we just ask it over HTTP.
    if health.get("collection_count", 0) > 0 and health.get("chroma_reachable"):
        st.divider()
        st.subheader("Distribution par source")
        stats = api_get("/index/stats")
        if stats:
            _render_source_distribution(stats)
        else:
            st.warning("Impossible de récupérer les stats d'index.")

        st.divider()
        st.subheader("Échantillon de chunks (5 premiers)")
        if stats:
            _render_sample_chunks(stats)


__all__ = ["render"]

"""Streamlit sidebar — system status + "About this tool" section.

Shown on every page (streamlit always renders the sidebar in the
same script run). The sidebar is shared by all four tabs, so it
lives outside of the per-tab modules to avoid duplication.
"""

from __future__ import annotations

import streamlit as st

from src.config import settings
from src.ui.api_client import api_get

_ABOUT_MARKDOWN = (
    "**Industrial Knowledge Copilot** est un copilote RAG local qui répond à des "
    "questions sur la **maintenance des roulements** (capacité de charge, lubrification, "
    "montage, diagnostic vibratoire, modes de défaillance) en s'appuyant sur :\n"
    "- les **catalogues Schaeffler, SKF et NTN-SNR** (≈ 5 000 pages, FR + EN)\n\n"
    "**À quoi il répond** : capacité de charge (C, C0, L10), lubrification (graisse, huile, "
    "intervalles), procédures de montage / démontage, diagnostic vibratoire, "
    "limites de température, modes de défaillance courants.\n\n"
    "**À quoi il ne répond pas** : questions hors domaine (météo, cuisine, "
    "prévisions boursières, courses de chevaux, etc.) — il le dit explicitement. "
    "Il ne fait pas non plus de calculs RUL ou d'analyse de séries temporelles "
    "de capteurs (pas de données structurées dans ce projet)."
)


def _render_about() -> None:
    """The "❓ À propos de cet outil" expander — visible to recruiters / first-time users."""
    with st.expander("❓ À propos de cet outil", expanded=True):
        st.markdown(_ABOUT_MARKDOWN)


def _render_system_status() -> None:
    """3-line JSON block: API status, chroma, mlx_ready, chunk count, hardware."""
    st.subheader("État du système")
    health = api_get("/health")
    if not health:
        st.warning(f"API injoignable sur :{settings.api_port}. Lance `make api` dans un terminal.")
        return
    st.json(
        {
            "statut": health["status"],
            "chroma": health["chroma_reachable"],
            "mlx_ready": health["mlx_ready"],
            "chunks_indexés": health["collection_count"],
            "hardware": health["hardware"],
        }
    )


def _render_footer() -> None:
    """Two small captions with the ports + the Apple Silicon requirement."""
    st.caption(f"Streamlit sur :{settings.ui_port} · API sur :{settings.api_port}")
    st.caption("MLX nécessite Apple Silicon (M1/M2/M3/M4/M5).")


def render() -> None:
    """Render the sidebar (called once, before the tabs)."""
    with st.sidebar:
        st.title("🛠️ IKC")
        st.caption("Copilote RAG local pour la maintenance industrielle.")
        _render_about()
        st.divider()
        _render_system_status()
        st.divider()
        _render_footer()


__all__ = ["render"]

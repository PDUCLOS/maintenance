"""RAGAS tab — show the latest eval snapshot + history of all snapshots.

Snapshots are immutable JSON files written by `make eval` into
`reports/eval_<timestamp>.json`. We read the filesystem directly
(no API roundtrip) — the API doesn't expose this anyway.

Each snapshot has:
  - timestamp_utc
  - n_samples
  - metrics: list[{name, score}]

We surface:
  - the latest snapshot with one st.metric per metric,
  - the full history as a DataFrame (newest first).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st


def _load_snapshot(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_color(score: float) -> str:
    if score >= 0.75:
        return "🟢"
    if score >= 0.5:
        return "🟡"
    return "🔴"


def render(reports_dir: Path) -> None:
    """Render the RAGAS tab content (called inside `with tab_ragas:`).

    `reports_dir` is the directory where `make eval` writes snapshots
    (defaults to `<project>/reports/`). It's passed in to keep the
    module pure-Python (no PROJECT_ROOT computation at import time)
    and trivially testable with a tmp_path.
    """
    st.title("📊 Évaluation RAGAS")
    st.caption(
        "Métriques RAGAS (faithfulness, answer relevancy, context precision, "
        "context recall). Chaque `make eval` crée un **snapshot immuable** dans "
        "`reports/`. Le plus récent est affiché ici, avec l'historique complet."
    )

    if not reports_dir.is_dir():
        st.info("Aucun snapshot. Lance `make eval` pour générer le premier.")
        return

    snapshots = sorted(reports_dir.glob("eval_*.json"), reverse=True)
    if not snapshots:
        st.info("Aucun snapshot. Lance `make eval` pour générer le premier.")
        return

    # --- Latest snapshot ---------------------------------------------------
    latest = snapshots[0]
    st.subheader(f"Dernier snapshot : `{latest.name}`")
    data = _load_snapshot(latest)
    ts = data.get("timestamp_utc", "?")
    n = data.get("n_samples", "?")
    st.caption(f"Date : {ts} · Échantillons : {n}")

    metrics = data.get("metrics", [])
    if metrics:
        cols = st.columns(len(metrics))
        for i, m in enumerate(metrics):
            name = m["name"]
            score = m["score"]
            with cols[i]:
                st.metric(
                    label=f"{_metric_color(score)} {name}",
                    value=f"{score:.3f}",
                )
        st.caption(
            "Cibles : faithfulness > 0.75, answer_relevancy > 0.75, "
            "context_precision > 0.70, context_recall > 0.65"
        )

    # --- History -----------------------------------------------------------
    if len(snapshots) > 1:
        st.divider()
        st.subheader("Historique")
        history = []
        for snap in snapshots:
            d = _load_snapshot(snap)
            row = {
                "snapshot": snap.name,
                "date_utc": d.get("timestamp_utc", "?"),
                "échantillons": d.get("n_samples", "?"),
            }
            for m in d.get("metrics", []):
                row[m["name"]] = round(m["score"], 3)
            history.append(row)
        st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


__all__ = ["_load_snapshot", "_metric_color", "render"]

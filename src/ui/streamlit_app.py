"""Streamlit UI — multi-tab interface for the RAG copilot.

Tabs:
  1. Chat         — the core RAG chat interface (existing)
  2. Inventory    — data/raw inventory: Schaeffler + SKF + NTN-SNR PDF catalogues
  3. RAGAS        — latest evaluation snapshot + history of all snapshots
  4. Index        — ChromaDB stats: chunk count, source distribution, sample

Run with:
    make ui    # http://localhost:8501

Calls the FastAPI backend on :8000 for the chat. The other tabs read
the local filesystem + RAGAS snapshots directly (no API roundtrip).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.config import settings
from src.rag.intents import INTENTS, build_question

API_BASE = f"http://localhost:{settings.api_port}"
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = PROJECT_ROOT / "reports"


# --- API helpers -------------------------------------------------------------


def _api_get(path: str, timeout: float = 5.0) -> dict | None:
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Appel API échoué : {e}")
        return None


def _api_post(path: str, payload: dict, timeout: float = 60.0) -> dict | None:
    try:
        r = requests.post(f"{API_BASE}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except requests.RequestException as e:
        st.error(f"Appel API échoué : {e}")
        return None


# --- PDF inspection helpers -------------------------------------------------


def _pdf_metadata(pdf_path: Path) -> dict:
    """Extract PDF metadata via the `pdfinfo` shell tool (poppler-utils)."""
    if not pdf_path.is_file():
        return {"pages": "—", "title": "(file not found)", "size_mb": "—"}
    try:
        out = subprocess.check_output(
            ["pdfinfo", str(pdf_path)], stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8", errors="ignore")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {
            "pages": "?",
            "title": "(pdfinfo unavailable)",
            "size_mb": f"{pdf_path.stat().st_size / 1048576:.1f}",
        }
    info: dict = {
        "pages": "?",
        "title": "(no title)",
        "size_mb": f"{pdf_path.stat().st_size / 1048576:.1f}",
    }
    for line in out.splitlines():
        if line.startswith("Pages:"):
            info["pages"] = line.split(":", 1)[1].strip()
        elif line.startswith("Title:"):
            t = line.split(":", 1)[1].strip()
            if t:
                info["title"] = t
    return info


# --- Page config -------------------------------------------------------------
st.set_page_config(
    page_title="Industrial Knowledge Copilot",
    page_icon="🛠️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Sidebar (common) --------------------------------------------------------
with st.sidebar:
    st.title("🛠️ IKC")
    st.caption("Copilote RAG local pour la maintenance industrielle.")

    # À propos / What this answers — section très visible pour les
    # visiteurs (recruteurs, users) qui ouvrent le projet.
    with st.expander("❓ À propos de cet outil", expanded=True):
        st.markdown(
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
            "de capteurs (pas de données structurées dans ce projet).",
        )

    st.divider()
    st.subheader("État du système")

    health = _api_get("/health")
    if health:
        st.json(
            {
                "statut": health["status"],
                "chroma": health["chroma_reachable"],
                "mlx_ready": health["mlx_ready"],
                "chunks_indexés": health["collection_count"],
                "hardware": health["hardware"],
            }
        )
    else:
        st.warning("API injoignable sur :8000. Lance `make api` dans un terminal.")

    st.divider()
    st.caption(f"Streamlit sur :{settings.ui_port} · API sur :{settings.api_port}")
    st.caption("MLX nécessite Apple Silicon (M1/M2/M3/M4/M5).")

# --- Main tabs ---------------------------------------------------------------
tab_chat, tab_inventory, tab_ragas, tab_index = st.tabs(
    ["💬 Chat", "📦 Inventaire", "📊 RAGAS", "🗂️ Index"]
)


# ============================================================================
# Tab 1: Chat
# ============================================================================
with tab_chat:
    st.title("Copilote de maintenance des roulements")
    st.caption(
        "Pose une question sur la **maintenance des roulements** — capacité de charge, "
        "lubrification, montage, diagnostic vibratoire, modes de défaillance. "
        "Les réponses sont construites à partir des catalogues Schaeffler, SKF et NTN-SNR. "
        "Tu peux écrire librement ou utiliser le **formulaire guidé** ci-dessous."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Controls above the chat history
    top_k = st.slider("Nombre de sources récupérées (top-K)", min_value=1, max_value=20, value=5)

    # --- Guided question (intent picker) -----------------------------------
    with st.expander("🎯 Question guidée — choisis un template", expanded=False):
        st.caption(
            "Si tu ne sais pas quoi demander, choisis un template dans la liste et "
            "remplis les champs. Le formulaire génère une question propre, bien formée, "
            "optimisée pour la base de connaissances."
        )
        # Group intents by category for a clean selectbox
        intent_labels_by_cat: dict[str, list[str]] = {}
        intent_by_label: dict[str, object] = {}
        for intent in INTENTS:
            label = f"{intent.icon}  {intent.label}  ·  ({intent.category.value})"
            intent_labels_by_cat.setdefault(intent.category.value, []).append(label)
            intent_by_label[label] = intent

        # Build a flat selectbox, grouped by category
        all_labels: list[str] = []
        for cat in intent_labels_by_cat:
            all_labels.append(f"── {cat} ──")
            all_labels.extend(intent_labels_by_cat[cat])
        all_labels.insert(0, "— (question libre plus bas) —")

        chosen_label = st.selectbox("Type de question", all_labels, index=0)
        chosen_intent = intent_by_label.get(chosen_label)

        if chosen_intent is not None:
            st.caption(chosen_intent.description)
            # Render one widget per field
            field_values: dict = {}
            for f in chosen_intent.fields:
                if f.kind == "select":
                    options = [v for v, _ in f.options] or [f.default]
                    labels = [lab for _, lab in f.options] or options
                    default_idx = options.index(f.default) if f.default in options else 0
                    field_values[f.name] = st.selectbox(
                        f.label, options, index=default_idx,
                        format_func=lambda v, lab=labels, o=options: lab[o.index(v)] if v in o else v,
                        key=f"intent_{chosen_intent.key}_{f.name}",
                    )
                elif f.kind == "number":
                    field_values[f.name] = str(st.number_input(
                        f.label,
                        min_value=f.min_value or 1,
                        max_value=f.max_value or 9999,
                        value=int(f.default) if f.default else 100,
                        key=f"intent_{chosen_intent.key}_{f.name}",
                    ))
                else:  # text
                    field_values[f.name] = st.text_input(
                        f.label,
                        value=f.default,
                        placeholder=f.placeholder,
                        key=f"intent_{chosen_intent.key}_{f.name}",
                    )

            col_a, col_b = st.columns([1, 3])
            with col_a:
                generate = st.button("💡 Générer la question", use_container_width=True)
            if generate:
                try:
                    generated_q = build_question(chosen_intent.key, **field_values)
                    st.session_state.pending_question = generated_q
                    st.success(f"Question générée : _{generated_q}_")
                except ValueError as e:
                    st.error(str(e))

        # If a question is pending (either typed or generated), feed it
        # into the chat input flow on the next rerun.
        pending = st.session_state.get("pending_question")
        if pending:
            st.info(f"Question prête à envoyer : **{pending}**")
            if st.button("📤 Envoyer cette question"):
                st.session_state.pending_question_from_input = pending
                st.session_state.pending_question = None
                st.rerun()

    # Chat history
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

    # Input (free-form chat). The guided-question form above can also
    # push a question via st.session_state.pending_question.
    if prompt := (st.chat_input("Pose ta question…") or st.session_state.pop("pending_question_from_input", None)):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            t0 = time.perf_counter()
            result = _api_post(
                "/query",
                {"question": prompt, "top_k": top_k, "stream": False},
            )
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
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )


# ============================================================================
# Tab 2: Inventory
# ============================================================================
with tab_inventory:
    st.title("📦 Inventaire des données")
    st.caption(
        "Provenance et métadonnées de toutes les sources ingérées. "
        "Voir `data/raw/pdf/INVENTORY.md` pour les détails complets."
    )

    # PDF section
    st.subheader("Catalogues industriels (Schaeffler, SKF, NTN-SNR)")
    pdf_dir = settings.pdf_dir
    if pdf_dir.is_dir():
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            st.info("Aucun PDF. Ajoutes-en dans `data/raw/pdf/` et relance `make ingest`.")
        else:
            pdf_data = []
            for f in pdfs:
                meta = _pdf_metadata(f)
                # Brand inference from filename
                name_lc = f.name.lower()
                if "schaeffler" in name_lc or "fag" in name_lc or "ina" in name_lc:
                    brand = "Schaeffler"
                elif "skf" in name_lc:
                    brand = "SKF"
                elif "ntn" in name_lc or "snr" in name_lc:
                    brand = "NTN-SNR"
                else:
                    brand = "?"
                pdf_data.append(
                    {
                        "Fichier": f.name,
                        "Marque": brand,
                        "Pages": meta["pages"],
                        "Taille": f"{meta['size_mb']} MB",
                        "Titre (métadonnée PDF)": meta["title"][:80]
                        + ("…" if len(meta["title"]) > 80 else ""),
                    }
                )
            st.dataframe(pd.DataFrame(pdf_data), use_container_width=True, hide_index=True)
            total_mb = sum(f.stat().st_size for f in pdfs) / 1048576
            st.caption(
                f"**Total : {len(pdfs)} PDFs, {total_mb:.1f} Mo.** "
                f"Source : sites officiels des fabricants (Schaeffler, SKF). "
                f"Inventaire détaillé : [`data/raw/pdf/INVENTORY.md`](data/raw/pdf/INVENTORY.md)."
            )
    else:
        st.warning(f"Dossier PDF manquant : {pdf_dir}.")


# ============================================================================
# Tab 3: RAGAS
# ============================================================================
with tab_ragas:
    st.title("📊 Évaluation RAGAS")
    st.caption(
        "Métriques RAGAS (faithfulness, answer relevancy, context precision, "
        "context recall). Chaque `make eval` crée un **snapshot immuable** dans "
        "`reports/`. Le plus récent est affiché ici, avec l'historique complet."
    )

    if not REPORTS_DIR.is_dir():
        st.info("Aucun snapshot. Lance `make eval` pour générer le premier.")
    else:
        snapshots = sorted(REPORTS_DIR.glob("eval_*.json"), reverse=True)
        if not snapshots:
            st.info("Aucun snapshot. Lance `make eval` pour générer le premier.")
        else:
            # Latest snapshot
            latest = snapshots[0]
            st.subheader(f"Dernier snapshot : `{latest.name}`")
            data = json.loads(latest.read_text(encoding="utf-8"))
            ts = data.get("timestamp_utc", "?")
            n = data.get("n_samples", "?")
            st.caption(f"Date : {ts} · Échantillons : {n}")

            # Metric cards
            metrics = data.get("metrics", [])
            if metrics:
                cols = st.columns(len(metrics))
                for i, m in enumerate(metrics):
                    name = m["name"]
                    score = m["score"]
                    with cols[i]:
                        # Color-code
                        color = "🟢" if score >= 0.75 else "🟡" if score >= 0.5 else "🔴"
                        st.metric(
                            label=f"{color} {name}",
                            value=f"{score:.3f}",
                        )
                st.caption(
                    "Cibles : faithfulness > 0.75, answer_relevancy > 0.75, "
                    "context_precision > 0.70, context_recall > 0.65"
                )

            # History table
            if len(snapshots) > 1:
                st.divider()
                st.subheader("Historique")
                history = []
                for snap in snapshots:
                    d = json.loads(snap.read_text(encoding="utf-8"))
                    row = {
                        "snapshot": snap.name,
                        "date_utc": d.get("timestamp_utc", "?"),
                        "échantillons": d.get("n_samples", "?"),
                    }
                    for m in d.get("metrics", []):
                        row[m["name"]] = round(m["score"], 3)
                    history.append(row)
                st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ============================================================================
# Tab 4: Index
# ============================================================================
with tab_index:
    st.title("🗂️ Index ChromaDB")
    st.caption(
        "État de l'index vectoriel en temps réel. Si la collection est vide, lance "
        "`make ingest` après avoir populé `data/raw/`."
    )

    health = _api_get("/health")
    if not health:
        st.warning("API injoignable sur :8000. Lance `make api`.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Chunks totaux", health.get("collection_count", 0))
        with col2:
            st.metric("Chroma joignable", "✅" if health.get("chroma_reachable") else "❌")
        with col3:
            st.metric("Hardware", health.get("hardware", "?"))

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
            stats = _api_get("/index/stats")
            if stats and stats.get("source_counts"):
                df_sources = pd.DataFrame(
                    [{"source": k, "count": v} for k, v in stats["source_counts"].items()]
                ).sort_values("count", ascending=False)
                st.bar_chart(df_sources.set_index("source"))
                st.caption(
                    f"Distribution sur {sum(stats['source_counts'].values())} chunks échantillonnés "
                    f"(plafond 10k pour la perf)."
                )
            elif stats:
                st.info("La collection est vide.")

            st.divider()
            st.subheader("Échantillon de chunks (5 premiers)")
            if stats and stats.get("sample_chunks"):
                for chunk in stats["sample_chunks"]:
                    with st.expander(f"`{chunk['id']}`", expanded=False):
                        st.caption(f"Source : `{chunk['source']}` · Métadonnées : `{chunk['metadata']}`")
                        text = chunk["text"]
                        st.text(text[:600] + ("…" if len(text) > 600 else ""))

"""Streamlit UI — multi-tab interface for the RAG copilot.

Tabs:
  1. Chat         — the core RAG chat interface (existing)
  2. Inventory    — data/raw inventory: CMAPSS subsets + Schaeffler + SKF PDFs
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
from datetime import datetime
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

from src.config import settings

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


# --- PDF / CMAPSS inspection helpers -----------------------------------------

def _pdf_metadata(pdf_path: Path) -> dict:
    """Extract PDF metadata via the `pdfinfo` shell tool (poppler-utils)."""
    if not pdf_path.is_file():
        return {"pages": "—", "title": "(file not found)", "size_mb": "—"}
    try:
        out = subprocess.check_output(
            ["pdfinfo", str(pdf_path)], stderr=subprocess.DEVNULL, timeout=10
        ).decode("utf-8", errors="ignore")
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return {"pages": "?", "title": "(pdfinfo unavailable)", "size_mb": f"{pdf_path.stat().st_size / 1048576:.1f}"}
    info: dict = {"pages": "?", "title": "(no title)", "size_mb": f"{pdf_path.stat().st_size / 1048576:.1f}"}
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
    st.caption("Local RAG copilot for industrial maintenance.")

    health = _api_get("/health")
    if health:
        st.write("**System status**")
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
    st.caption(f"Streamlit on :{settings.ui_port} · API on :{settings.api_port}")
    st.caption(f"MLX requires Apple Silicon (M1/M2/M3/M4/M5)")

# --- Main tabs ---------------------------------------------------------------
tab_chat, tab_inventory, tab_ragas, tab_index = st.tabs(
    ["💬 Chat", "📦 Inventory", "📊 RAGAS", "🗂️ Index"]
)


# ============================================================================
# Tab 1: Chat
# ============================================================================
with tab_chat:
    st.title("Industrial Knowledge Copilot")
    st.caption(
        "Pose une question sur la maintenance industrielle (NASA CMAPSS) "
        "ou sur les catalogues techniques (Schaeffler, SKF)."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Controls above the chat history
    c1, c2 = st.columns([3, 1])
    with c1:
        top_k = st.slider("Top-K retrieved chunks", min_value=1, max_value=20, value=5)
    with c2:
        use_agent = st.toggle(
            "Use agent (Python tool calling)",
            value=False,
            help="Allows the copilot to run pandas queries on CMAPSS data.",
        )

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

    # Input
    if prompt := st.chat_input("Ta question…"):
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
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })


# ============================================================================
# Tab 2: Inventory
# ============================================================================
with tab_inventory:
    st.title("📦 Data inventory")
    st.caption(
        "Provenance et métadonnées de toutes les sources ingérées. "
        "Voir `data/raw/pdf/INVENTORY.md` pour les détails."
    )

    # CMAPSS section
    st.subheader("NASA CMAPSS (turbofan engine degradation)")
    cmapss_dir = settings.cmapss_dir
    if cmapss_dir.is_dir():
        cmapss_files = sorted(cmapss_dir.glob("*.txt")) + sorted(cmapss_dir.glob("*.pdf"))
        cmapss_data = []
        for f in cmapss_files:
            if f.suffix == ".txt":
                size_kb = f.stat().st_size / 1024
                cmapss_data.append({
                    "File": f.name,
                    "Type": "Text data",
                    "Size": f"{size_kb:.0f} KB",
                })
            elif f.suffix == ".pdf":
                meta = _pdf_metadata(f)
                cmapss_data.append({
                    "File": f.name,
                    "Type": f"PDF ({meta['pages']} pages)",
                    "Size": f"{meta['size_mb']} MB",
                })
        st.dataframe(pd.DataFrame(cmapss_data), use_container_width=True, hide_index=True)
    else:
        st.warning(f"CMAPSS directory missing: {cmapss_dir}. Run `make data`.")

    st.divider()

    # PDF section
    st.subheader("Industrial catalogues (Schaeffler + SKF)")
    pdf_dir = settings.pdf_dir
    if pdf_dir.is_dir():
        pdfs = sorted(pdf_dir.glob("*.pdf"))
        if not pdfs:
            st.info("No PDFs yet. Add some to `data/raw/pdf/` and re-run `make ingest`.")
        else:
            pdf_data = []
            for f in pdfs:
                meta = _pdf_metadata(f)
                # Brand inference from filename
                brand = "Schaeffler" if "schaeffler" in f.name.lower() or "fag" in f.name.lower() else "SKF" if "skf" in f.name.lower() else "?"
                pdf_data.append({
                    "File": f.name,
                    "Brand": brand,
                    "Pages": meta["pages"],
                    "Size": f"{meta['size_mb']} MB",
                    "Title (PDF metadata)": meta["title"][:80] + ("…" if len(meta["title"]) > 80 else ""),
                })
            st.dataframe(pd.DataFrame(pdf_data), use_container_width=True, hide_index=True)
            total_mb = sum(f.stat().st_size for f in pdfs) / 1048576
            st.caption(
                f"**Total: {len(pdfs)} PDFs, {total_mb:.1f} MB.** "
                f"Source: official manufacturer sites (Schaeffler, SKF). "
                f"Detailed inventory: [`data/raw/pdf/INVENTORY.md`](data/raw/pdf/INVENTORY.md)."
            )
    else:
        st.warning(f"PDF directory missing: {pdf_dir}.")


# ============================================================================
# Tab 3: RAGAS
# ============================================================================
with tab_ragas:
    st.title("📊 RAGAS evaluation")
    st.caption(
        "Métriques RAGAS (faithfulness, answer relevancy, context precision, "
        "context recall). Chaque `make eval` crée un snapshot immutable dans "
        "`reports/`. Le plus récent est affiché ici, avec l'historique."
    )

    if not REPORTS_DIR.is_dir():
        st.info("No snapshots yet. Run `make eval` to generate the first one.")
    else:
        snapshots = sorted(REPORTS_DIR.glob("eval_*.json"), reverse=True)
        if not snapshots:
            st.info("No snapshots yet. Run `make eval` to generate the first one.")
        else:
            # Latest snapshot
            latest = snapshots[0]
            st.subheader(f"Latest snapshot: `{latest.name}`")
            data = json.loads(latest.read_text(encoding="utf-8"))
            ts = data.get("timestamp_utc", "?")
            n = data.get("n_samples", "?")
            st.caption(f"Timestamp: {ts} · Samples: {n}")

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
                    "Targets: faithfulness > 0.75, answer_relevancy > 0.75, "
                    "context_precision > 0.70, context_recall > 0.65"
                )

            # History table
            if len(snapshots) > 1:
                st.divider()
                st.subheader("History")
                history = []
                for snap in snapshots:
                    d = json.loads(snap.read_text(encoding="utf-8"))
                    row = {"snapshot": snap.name, "timestamp_utc": d.get("timestamp_utc", "?"), "n_samples": d.get("n_samples", "?")}
                    for m in d.get("metrics", []):
                        row[m["name"]] = round(m["score"], 3)
                    history.append(row)
                st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


# ============================================================================
# Tab 4: Index
# ============================================================================
with tab_index:
    st.title("🗂️ ChromaDB index")
    st.caption(
        "État de l'index vectoriel. Si la collection est vide, lance "
        "`make ingest` après avoir populé `data/raw/`."
    )

    health = _api_get("/health")
    if not health:
        st.warning("API unreachable on :8000. Start it with `make api`.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total chunks", health.get("collection_count", 0))
        with col2:
            st.metric("Chroma reachable", "✅" if health.get("chroma_reachable") else "❌")
        with col3:
            st.metric("Hardware", health.get("hardware", "?"))

        # Source distribution
        if health.get("collection_count", 0) > 0 and health.get("chroma_reachable"):
            st.divider()
            st.subheader("Source distribution")
            try:
                # Direct Chroma call (we're in the same docker network conceptually,
                # but actually we're on the host, so we use HTTP)
                import chromadb

                client = chromadb.HttpClient(
                    host=settings.chroma_host, port=settings.chroma_port
                )
                collection = client.get_or_create_collection(settings.chroma_collection)
                data = collection.get(include=["metadatas"], limit=10000)
                metas = data.get("metadatas") or []
                if metas:
                    source_counts: dict[str, int] = {}
                    for m in metas:
                        src = (m or {}).get("source", "unknown")
                        # Truncate long source names for the chart
                        short = src.split(":")[0] if ":" in src else src
                        source_counts[short] = source_counts.get(short, 0) + 1
                    df_sources = pd.DataFrame(
                        [{"source": k, "count": v} for k, v in source_counts.items()]
                    ).sort_values("count", ascending=False)
                    st.bar_chart(df_sources.set_index("source"))
                    st.caption(
                        f"Distribution over {len(metas)} sampled chunks "
                        f"(cap at 10k for performance)."
                    )
                else:
                    st.info("Collection is empty.")
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not query Chroma: {e}")

            st.divider()
            st.subheader("Sample chunks (first 5)")
            try:
                data = collection.get(limit=5, include=["documents", "metadatas"])
                for i, doc_id in enumerate(data.get("ids") or []):
                    with st.expander(f"`{doc_id}`", expanded=False):
                        text = data["documents"][i] if data.get("documents") else ""
                        meta = data["metadatas"][i] if data.get("metadatas") else {}
                        st.caption(f"Source: `{meta.get('source', '?')}` · Metadata: `{meta}`")
                        st.text(text[:600] + ("…" if len(text) > 600 else ""))
            except Exception as e:  # noqa: BLE001
                st.warning(f"Could not fetch sample chunks: {e}")

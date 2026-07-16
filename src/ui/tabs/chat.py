"""Chat tab — the core RAG chat interface.

Responsibilities:
  - session-state for the message history and pending question,
  - the top-K slider (over-fetch knob),
  - the guided-question form (intent picker, 10 bearing intents),
  - the chat history rendering with sources expander,
  - the chat input + assistant response loop.

The intent-picker form reads `INTENTS` from `src.rag.intents` so
adding a new guided question only touches one place (intents.py).
"""

from __future__ import annotations

import time

import streamlit as st

from src.rag.intents import INTENTS, build_question
from src.ui.api_client import api_post


def _render_guided_question_form() -> None:
    """Render the intent-picker expander and return the chosen question, if any.

    Returns None if the user hasn't picked an intent or hasn't clicked
    "Générer la question". The caller decides whether to push the
    result into the chat input.
    """
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
                        f.label,
                        options,
                        index=default_idx,
                        format_func=lambda v, lab=labels, o=options: lab[o.index(v)]
                        if v in o
                        else v,
                        key=f"intent_{chosen_intent.key}_{f.name}",
                    )
                elif f.kind == "number":
                    field_values[f.name] = str(
                        st.number_input(
                            f.label,
                            min_value=f.min_value or 1,
                            max_value=f.max_value or 9999,
                            value=int(f.default) if f.default else 100,
                            key=f"intent_{chosen_intent.key}_{f.name}",
                        )
                    )
                else:  # text
                    field_values[f.name] = st.text_input(
                        f.label,
                        value=f.default,
                        placeholder=f.placeholder,
                        key=f"intent_{chosen_intent.key}_{f.name}",
                    )

            col_a, _ = st.columns([1, 3])
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


def _render_sources(sources: list[dict]) -> None:
    """Render the source list under an expander (one row per chunk)."""
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})", expanded=False):
        for s in sources:
            text = s["text"]
            st.markdown(
                f"**`{s['id']}`** · `{s['source']}` · score={s['score']:.3f}\n\n"
                f"> {text[:400]}{'…' if len(text) > 400 else ''}"
            )


def _run_query(prompt: str, top_k: int) -> tuple[str, list[dict], float]:
    """Call /query and unpack the (answer, sources, latency_ms) tuple.

    Returns a placeholder answer when the API is unreachable — the
    chat history stays coherent and the user sees a clear error.
    """
    t0 = time.perf_counter()
    result = api_post("/query", {"question": prompt, "top_k": top_k, "stream": False})
    elapsed_ms = (time.perf_counter() - t0) * 1000
    if result is None:
        return "❌ L'API n'a pas répondu.", [], elapsed_ms
    answer = result.get("answer", "(pas de réponse)")
    sources = result.get("sources", [])
    return answer, sources, elapsed_ms


def render() -> None:
    """Render the Chat tab content (called inside `with tab_chat:`)."""
    st.title("Copilote de maintenance des roulements")
    st.caption(
        "Pose une question sur la **maintenance des roulements** — capacité de charge, "
        "lubrification, montage, diagnostic vibratoire, modes de défaillance. "
        "Les réponses sont construites à partir des catalogues Schaeffler, SKF et NTN-SNR. "
        "Tu peux écrire librement ou utiliser le **formulaire guidé** ci-dessous."
    )

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # --- Welcome prompt (empty chat) ---------------------------------------
    if not st.session_state.messages:
        st.info(
            "👋 **Bienvenue !** Pose ta question en français ou en anglais — "
            "le copilote répond dans la même langue que toi, et cite ses sources.\n\n"
            "**Si tu ne sais pas par où commencer, essaie :**\n"
            "- _« Quelle est la capacité de charge dynamique de base (C) d'un roulement ? »_\n"
            "- _« What is the rating life (L10) of a rolling bearing? »_\n"
            "- _« Comment choisir une graisse pour un roulement à billes ? »_\n"
            "- _« What are the most common failure modes for rolling bearings? »_\n\n"
            "Ou utilise le **formulaire guidé** ci-dessous pour poser une question structurée."
        )

    # --- Controls above the chat history ----------------------------------
    top_k = st.slider("Nombre de sources récupérées (top-K)", min_value=1, max_value=20, value=5)

    # --- Guided-question form ---------------------------------------------
    _render_guided_question_form()

    # --- Chat history ------------------------------------------------------
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_sources(msg.get("sources", []))

    # --- Chat input (free-form OR pending guided question) ----------------
    if prompt := (
        st.chat_input("Pose ta question…")
        or st.session_state.pop("pending_question_from_input", None)
    ):
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            answer, sources, elapsed_ms = _run_query(prompt, top_k)
            st.markdown(answer)
            st.caption(f"⏱️ {elapsed_ms:.0f} ms · {len(sources)} sources")
            _render_sources(sources)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sources": sources,
                }
            )


__all__ = ["render"]

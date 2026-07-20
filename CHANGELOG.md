# Changelog

All notable changes to **Industrial Knowledge Copilot** are documented
in this file. The format is based on [Keep a Changelog](https://keepachangelog.com/),
and this project adheres to [Semantic Versioning](https://semver.org/) (loosely,
as we're at 0.x.y and the API surface is still in flux).

The dates are in `YYYY-MM-DD` format. Commits are referenced by their
short SHA (first 7 chars) for traceability.

---

## [Unreleased]

### Added
- **Hard timeout on /query endpoint** — `chain.query()` is wrapped in a
  `ThreadPoolExecutor` with `Future.result(timeout=...)`. If the LLM takes
  longer than `QUERY_TIMEOUT_SECONDS` (default 60s), the API returns 504
  with a clear error body. Configurable via env var.
- **`/query` smoke test** — `scripts/test_query_relevance.py` validates 6
  representative questions for relevance (does the answer address the
  question?) and jargon hygiene (does the answer contain developer
  words like "chunk", "retrieval", "embedding"?).
- **Ingest progress bar** — `tqdm` is now shown during the bge-m3 embed
  (which takes ~20 min on M5 Pro). Bar only appears for batches > 50 chunks,
  so single-query RAG stays quiet.
- **Pydantic v2 config in MLXChatModel** — replaced deprecated
  `class Config: arbitrary_types_allowed = True` with
  `model_config = ConfigDict(arbitrary_types_allowed=True)`.

### Changed
- **`bge-small-en-v1.5` → `bge-m3`** (multilingual, 1024-dim). The previous
  model was English-only and broke FR queries. bge-m3 is slower on MPS
  (~20 min for 11k chunks vs ~2 min for bge-small) but supports a single
  vector space for FR + EN, which the project needs.
- **`Mistral-7B` → `Qwen2.5-7B-Instruct`** (still 4-bit, still MLX). A/B
  test on the ReAct agent showed Qwen2.5 5/5 clean tool invocations vs
  Mistral 2/5. Same 4-bit/7B footprint.
- **`/query` endpoint** — now returns `504` on timeout and `500` with a
  clean body (no traceback) on internal error. `latency_ms` is included
  in both success and error responses.
- **FR/EN language detection** — added common French function words
  (je, tu, ne, pas, sais, etc.) to the hint set. Short FR queries like
  "Je ne sais pas" now correctly classify as FR (was EN before).

### Removed
- **CMAPSS dataset and tooling** (commit `360ae4a`):
  - `src/ingestion/cmapss_loader.py` (deleted)
  - `src/rag/agent.py` — closed-DSL tool now a stub (raises `NotImplementedError`)
  - `src/eval/dataset.py` — switched from CMAPSS-derived Q&A to PDF-derived
  - `src/rag/intents.py` — CMAPSS intents replaced with bearing-catalog intents
  - `data/datasets/` (45 Go of CWRU, MFPT, Paderborn, XJTU-SY, FEMTO-ST, NASA IMS)
  - `data/raw/cmapss/` (43 Mo of CMAPSS .txt + 1 PDF)
- **Old smoke test cases** (mirror_en: "How many sensors in CMAPSS?") —
  no longer relevant after the corpus pivot.

### Security
- **AI agent attribution removed from drawio XML** (commit `0cf52c4`):
  - The `agent="Mavis/0.1"` attribute in `docs/diagrams/pipeline.drawio`
    was a hidden Mavis signature. Stripped.
  - `docs/pipeline.md` §7 (commit policy) updated to forbid
    `agent="..."` in XML/JSON/DrawIO files (verified with
    `git grep -E 'agent="(Mavis|Mavis|Claude|Cursor|Copilot)"'`).
- **No AI attribution in commit history** — all commits have:
  - Author: `Patrice Duclos <patrice@lyonflow.fr>`
  - No `Co-Authored-By: <AI>` footer
  - No `🤖 Generated with <AI>` footer
  - Verified across 22 commits.

### Added
- **Streamlit UI split into per-tab modules** (commit `1014cb8`):
  - `src/ui/api_client.py` — httpx wrapper (`api_get` / `api_post`),
    streamlit-agnostic, base URL hardcoded to `localhost:<port>`.
  - `src/ui/sidebar.py` — common sidebar (about + system status).
  - `src/ui/tabs/{chat,inventory,ragas,index}.py` — one render()
    function per tab.
  - `src/ui/streamlit_app.py` — 52-line thin shell, was 485 lines.
- **18 new UI unit tests** (`tests/test_ui_helpers.py`): api_get/post
  (4 error paths + 2 happy paths), base URL, brand inference
  (Schaeffler / FAG-INA / SKF / NTN-SNR / case-insensitive),
  metric coloring (>=0.75 green, >=0.5 yellow, else red), snapshot
  loading.
- **One-command dev launcher** — `make dev` → `scripts/dev_up.sh`
  boots ChromaDB (Docker) + FastAPI + Streamlit with `Ctrl+C` to stop.
  Sets `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` / `PYTHONUNBUFFERED=1`
  for stable local runs.

### Changed
- **Versioned API namespace `/v1/*`** (commit `a87b512`): the FastAPI
  app now exposes both the legacy bare paths (`/query`, `/ingest`,
  `/eval`, `/health`, `/index/stats`) AND the new `/v1/*` prefix
  pointing to the same handlers. Backward compat: existing clients
  that hit `/query` keep working. New clients should use `/v1/query`.
- **Hard timeout on `/query`** (commit `4b2fcbc`): the LLM call is
  wrapped in a `ThreadPoolExecutor` (max_workers=4). 60s default,
  returns 504 with `{"error": "query_timeout", "elapsed_ms": N}` on
  exceed, 500 with `{"error": "internal_error"}` on raise.
- **Per-source retrieval precision** (commit `0b54120`): custom
  RAGAS metric aggregating `expected_source` hits per PDF filename.
  10/25 eval items have an `expected_source` annotation; the metric
  reports one number per source so we can see which catalogue the
  retriever surfaces most reliably.
- **CHANGELOG references** — the document now cross-references commit
  short-SHAs for every change (instead of just "what", it shows
  "where in git history").

### Fixed
- **Ruff format drift on test files** — `tests/test_api.py` and
  `tests/test_pure_helpers.py` reformatted to match the project's
  ruff config (line-length 100, pyupgrade rules).
- **Duplicate `top_k` slider in Streamlit** — a leftover copy of the
  `st.slider(...)` line in the chat tab was rendering the control
  twice. Removed.

### Added
- **Bilingual dataset generation (FR + EN)** — the dataset builder now
  detects the page's language (`src.rag.language.detect_language` on
  the full page text) and emits the question in that language. EN
  pages get EN templates; FR pages (the NTN-SNR catalogue +
  diagnostic guide) get FR templates with the FR equivalent of the
  topic. The dense retriever (bge-m3) is multilingual, but an EN
  query for "rating life" surfaces EN content (SKF/Schaeffler) and
  misses the same topic in FR (NTN-SNR) — generating the question
  in the page's own language fixes this and makes the per-source
  metric meaningful for the FR-only PDFs. Two new artefacts:
  - `TOPIC_FR` — canonical-name → "le / la / les / l' + noun" map
    (article baked into the topic so templates don't need to
    produce the article themselves — avoids "de le" elision bugs).
  - `QUESTION_TEMPLATES_FACTUAL/REASONING/RETRIEVAL` — now a
    `{en: [...], fr: [...]}` dict, both lists in the same template
    style ("Que dit cette page à propos de {topic} ?", "Pourquoi
    s'intéresse-t-on à {topic} dans la maintenance ?"). The FR
    reasoning templates use impersonal "s'intéresser à" / "il faut"
    forms to dodge gender agreement (FR verbs agree with their
    subject in m./f., s./pl., which is impossible to know at
    template-fill time).
- **Stratified sampling by PDF (`_sample_stratified_by_pdf`)** —
  pure random sampling from the corpus let the 4 dominant PDFs
  (Schaeffler + SKF, ~80% of pages) eat all 20 dataset slots, leaving
  the 9 smaller PDFs (NTN-SNR, GGB, FAG, sp1, etc.) with zero
  representation. Round-robin stratified sampling guarantees each
  PDF gets at least one question. With seed=42, NTN-SNR is now
  in the dataset for the first time (3 questions, all in French).

### Changed
- **TOPICS list bilingual** — each of the 20 topics now has a list
  of EN + FR synonyms. A topic is matched on a page if ANY synonym
  appears in the page text. Before this fix, NTN-SNR's FR-only
  pages couldn't match the EN-only TOPICS list and were dropped
  (1059 pages dropped, no NTN-SNR in the dataset). After: 1021
  dropped, NTN-SNR is in the dataset.
- **Pages tuple is now 4-element** — was `(source, paragraph,
  topics)`, now `(source, paragraph, full_text, topics)`. The
  full text is needed for accurate language detection (short
  paragraphs dominated by tables / bullet lists can be
  mis-classified). Memory cost: ~10 MB total for the ~4000-page
  corpus — fine.

### Tests
- 14 new unit tests in `tests/test_pure_helpers.py`:
  - 5 in `TestPageMatchingTopics` for the FR synonyms (lubrification,
    capacité de charge, montage, étanchéité, modes de défaillance).
  - 6 in `TestSampleStratifiedByPdf` (round-robin covers all PDFs,
    deterministic with seed, no duplicate page picks in one call,
    empty pool, n=0, real-shape distribution).
  - 7 in `TestFormatQuestion` (EN page → EN question, FR page → FR
    question with article, FR reasoning template has no broken
    gender agreement, FR retrieval template uses 'sur' to avoid
    'de le' elision, every FR topic has an article, the real
    dataset is bilingual).
- Total: 130 unit tests passing (was 114). Ruff clean. Two tests
  are skipped when chromadb isn't running on :8001 (they're
  integration-style — `TestDatasetTopicPageAlignment` and
  `TestBilingualRetrieverReachability`).

---

## [0.1.0] — 2026-07-16

Initial public release of the pivot (post-CMAPSS). Focus is now 100%
on the bearing-catalog RAG.

### Added
- **W0: Scaffold** (commit `52e95f2`) — repo skeleton, Makefile, CI,
  pyproject.toml, 13 PDF catalogues, dev environment.
- **W1: Ingestion** (commit `f68f7ab`) — CMAPSS + PDF loaders, recursive
  chunker, full pipeline. (CMAPSS path later removed in `360ae4a`.)
- **W2: RAG** (commit `b488cf8`) — hybrid retriever (BM25 + dense, RRF),
  LCEL chain, ReAct agent with `query_cmapss` tool. (Agent later
  stubbed in `360ae4a`.)
- **W3: Evaluation** (commit `6dbd597`) — RAGAS dataset generator
  (30 CMAPSS Q&A) and runner. (Dataset later regenerated from PDFs
  in `360ae4a`.)
- **W4: Reranker + docs** (commit `b3c86e4`) — cross-encoder reranker,
  README polish, 10 demo questions.
- **Web landing + demo** (commit `98ccd0f`) — `web/index.html` (60 KB
  professional landing) + `web/demo.html` (interactive chat demo).
- **Collection management** (commit `06c7809`) — `scripts/manage_collection.py`
  with `list/info/new/drop/use` subcommands, refuses to drop active
  collection, requires confirmation for destructive ops.
- **9 guided-question intents** (commit `5592ed5`) — dropdowns in the UI
  for common questions (C, L10, mounting, etc.).
- **100% French UI** (commit `6a76b66`) — every visible string in the
  Streamlit app is in French, with an "À propos de cet outil" section.
- **Welcome prompt + sample questions** (commit `1d2a47e`) — first-time
  users see 4 example questions (2 FR + 2 EN) in the chat.
- **PROJECT_OVERVIEW.docx** (commit `d6a0a9a`) — auto-generated Word
  document for portfolio reviewers.

### Fixed
- 5 critical bugs from code audit (commit `50916ec`):
  - `chain._format_context` returned empty string for empty chunks
  - `reranker.rerank` over-fetch was missing
  - `_resolve_sensor` didn't validate min/max
  - `_query_cmapss_impl` raised `KeyError` for invalid sensors
  - `_parse_dsl_query` couldn't handle multi-key dicts
- **Jargon hygiene in LLM answer** (commit `e139580`):
  - New rule 8 in `SYSTEM_PROMPT_MIRROR`: "Never use 'chunk', 'retrieval',
    'embedding', 'vector', 'RAG', 'passage', 'extrait' in the user-facing
    answer."
  - Context format in `chain._format_context` changed from
    `(source=..., retrieval=hybrid, score=0.87)` to
    `(source: ..., relevance: 0.87)` so the LLM doesn't echo the
    word "retrieval".
- **CI matrix: drop untested py3.13** (commit `5793947`): `torch==2.5.1`
  has no macOS py3.13 wheel; CI now only tests py3.12 (the version we
  actually run on).
- **Ruff format --check on 6 files** (commit `5793947`): reformatted
  files that had drifted from the formatter.
- **Dependabot config removed** (commit `f1c6cd1`): portfolio repo,
  no need for automated dependency-update PRs.

---

## Project context

This is a **portfolio project** built by Patrice Duclos (RNCP 38777
Lead Data / AI Architect) to demonstrate production-shaped RAG
engineering on a MacBook Pro M5 Pro with MLX. The pivot from
CMAPSS (aerospace) to 100% bearing catalogues (industrial maintenance)
was done in July 2026 to align the corpus with the project title.

For the A/B test (Qwen2.5 vs Mistral) and the original CMAPSS-based
work, see `git log` before commit `360ae4a`.

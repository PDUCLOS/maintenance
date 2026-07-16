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

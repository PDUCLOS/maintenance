# Industrial Knowledge Copilot

> Local RAG copilot for industrial maintenance knowledge — NASA CMAPSS turbofan
> degradation data + technical PDFs, running on Apple Silicon with MLX.

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](.github/workflows/ci.yml)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platform: Apple Silicon](https://img.shields.io/badge/platform-Apple%20Silicon-black)](https://support.apple.com/mac)

---

## What this is

A production-shaped RAG (Retrieval-Augmented Generation) system that answers
natural-language questions about industrial maintenance. It combines:

- **Document retrieval** over NASA CMAPSS technical documentation and
  industrial PDF catalogues (chunked + embedded in ChromaDB)
- **Tool calling** on a Python pandas DataFrame for quantitative questions
  (sensor stats, fleet size, RUL)
- **A local LLM** (Mistral 7B Instruct, 4-bit, MLX-quantized) running on
  Apple Silicon via Apple's MLX framework

100% local. No API key. No data egress. Built to be evaluated with RAGAS,
shipped with Docker Compose, and demoed in 5 minutes.

## Why it exists

This is a portfolio project designed to fill the **LLM/RAG/GenAI in
production** gap on my CV. The full pitch is in
[`docs/pitch_entrevue.md`](docs/pitch_entrevue.md).

## Quickstart

**Prerequisites:** macOS on Apple Silicon (M1/M2/M3/M4/M5), Python 3.12+,
Docker Desktop, a free [NASA PCoE account](https://ti.arc.nasa.gov/tech/dash/groups/pcoe/prognostic-data-repository/).

```bash
git clone https://github.com/PDUCLOS/industrial-knowledge-copilot
cd industrial-knowledge-copilot

make setup              # create venv, install deps
make pull-models        # download Mistral 7B + bge-small (~5 GB, one time)
make data               # instructions to download NASA CMAPSS
# ↑ manually drop the 4 train/test/RUL files into data/raw/cmapss/

make chroma-up          # start ChromaDB in Docker
make ingest             # build the vector index
make api                # start the API on :8000
make ui                 # start the Streamlit UI on :8501  (separate terminal)
```

Open <http://localhost:8501> and ask:

> *How many turbofan engines are in the FD001 training set?*

The copilot will answer in a few seconds, with the source chunks visible in
the sidebar.

## Architecture

```mermaid
flowchart LR
    UI[Streamlit UI<br/>:8501] -->|HTTP| API[FastAPI<br/>:8000]
    API -->|invoke| RAG[RAG Chain<br/>LangChain LCEL]
    RAG -->|embed| EMB[bge-small<br/>MPS]
    RAG -->|retrieve| HYB[Hybrid Retriever<br/>RRF]
    HYB -->|dense| CHR[(ChromaDB<br/>:8001)]
    HYB -->|BM25| BM25[(In-memory<br/>BM25 index)]
    RAG -->|generate| LLM[Mistral 7B<br/>MLX]
    RAG -->|tool| TOOL[query_cmapss<br/>pandas]
    TOOL --> DF[(CMAPSS<br/>DataFrame)]
```

Detailed diagrams and trade-offs: [`docs/architecture.md`](docs/architecture.md).

**Why MLX on the host, ChromaDB in Docker?** Docker Desktop on macOS runs in
a Linux/arm64 VM — Metal is not exposed there, so MLX would either fail or
silently fall back to slow CPU. We refuse that compromise. MLX runs natively
on the host (Metal direct), ChromaDB runs in Docker (no such constraint).
See `Makefile` for the orchestration.

## Stack

| Layer | Choice | Why |
|-------|--------|-----|
| LLM | Mistral 7B Instruct (4-bit, MLX) | Local, FR-correct, free, fast on M-series |
| Embeddings | bge-small-en-v1.5 (4-bit, MPS) | 33M params, fast, high-quality EN |
| Vector store | ChromaDB 0.5 | Local, simple, sufficient for < 100k chunks |
| Orchestration | LangChain v0.3 LCEL | Standard market 2026, requested in 80% of JDs |
| Hybrid retrieval | BM25 + dense (RRF) | Catches exact-match (sensor IDs) that embeddings miss |
| Evaluation | RAGAS v0.2 | Standard for RAG metrics |
| API | FastAPI + Uvicorn | Type-safe, fast, well-known |
| UI | Streamlit | Quick demo, easy to share |
| Container | Docker Compose (ChromaDB only) | Reproducible, no host pollution |

All versions pinned in [`requirements.txt`](requirements.txt).

## Evaluation (RAGAS)

We track **faithfulness, answer relevancy, context precision, context
recall** on a deterministic 30-item Q&A dataset generated from CMAPSS.
Methodology and tuning levers: [`docs/evaluation.md`](docs/evaluation.md).

```bash
make eval-dataset     # regenerate the 30 Q&A items (deterministic, seed=42)
make eval             # run RAGAS, snapshot to reports/eval_<UTC>.json
```

| Metric | W3 baseline | W4 target |
|--------|-------------|-----------|
| Faithfulness | ~0.50 | > 0.75 |
| Answer relevancy | ~0.60 | > 0.75 |
| Context precision | ~0.50 | > 0.70 |
| Context recall | ~0.45 | > 0.65 |
| Latency (M-series GPU) | ~5 s | < 5 s |

## Project layout

```
src/
├── config.py                # pydantic-settings, validates Apple Silicon
├── ingestion/               # CMAPSS + PDF loaders, chunker, pipeline
├── rag/                     # embeddings, vectorstore, retriever, MLX LLM, chain, agent
├── api/                     # FastAPI app + routes
├── ui/                      # Streamlit chat
├── eval/                    # RAGAS dataset + runner
└── utils/                   # loguru, timing

tests/                       # unit + integration (latter skipped on non-Mac)
docs/                        # architecture, evaluation, pitch
scripts/                     # setup_mlx, ingest, run_eval, clean
data/raw/                    # CMAPSS + PDFs (gitignored)
data/processed/              # chunks.jsonl, eval_dataset.jsonl (gitignored)
reports/                     # RAGAS snapshots (gitignored JSON)
```

## Development

```bash
make preflight         # verify Apple Silicon + Python 3.12 + Docker
make test              # unit tests only (no live services)
make test-integration  # full suite, requires Chroma + downloaded models
make test-cov          # with coverage report
make lint              # ruff
make format            # ruff auto-fix
```

## Why Apple Silicon only

MLX is Apple's own ML framework. It uses Metal (the macOS GPU API) and the
unified memory of Apple Silicon. There is **no CPU fallback, no other
backend** — the LLM module refuses to load on non-Apple-Silicon machines.
This is intentional: no silent degradation, no fake "it works in dev but
fails in prod" surprises.

If you need to run on Linux/cloud, the architecture supports swapping
MLX for a Transformers backend (vLLM, TGI, or OpenAI-compatible API). The
LLM adapter in `src/rag/llm.py` is the only place that needs to change.

## License

MIT — see [LICENSE](LICENSE).

## Author

**Patrice Duclos** — RNCP 38777 Lead Data / AI Architect
[LinkedIn](https://www.linkedin.com/in/patriceduclos/) ·
[CV](https://github.com/PDUCLOS/cv)

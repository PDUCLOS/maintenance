# =============================================================================
# Industrial Knowledge Copilot — Dockerfile
# =============================================================================
# IMPORTANT: This Dockerfile is for the API and UI services only.
# The LLM (MLX) MUST run natively on the macOS host because:
#   1. MLX requires Apple's Metal API (not available in Docker Desktop's Linux VM)
#   2. Docker on Mac runs in a Linux/arm64 VM, so MLX cannot access Metal
#
# ChromaDB runs in Docker (declared in docker-compose.yml).
# The API and UI run on the host (see Makefile) and connect to both.
#
# This Dockerfile is kept for:
#   - CI environments (running the API in a Linux container for tests)
#   - Linux / Kubernetes deployment with the portable backend: generation and
#     bge-m3 embeddings served by Ollama (GPU node), Chroma in server mode
#     (see deploy/k8s). The same image runs the API, the Streamlit UI and the
#     ingestion Job — only the command changes.
#
# Build: docker build -t maintenance-copilot:0.2.0 .
# =============================================================================

# syntax=docker/dockerfile:1.7

# ----- Stage 1: builder -----------------------------------------------------
FROM python:3.12-slim AS builder

# System deps for building wheels (numpy, pandas, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps in a venv so we can copy cleanly
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
# CPU-only torch wheel: torch is only used by the small cross-encoder reranker
# here (the GPU work happens in Ollama). The default PyPI Linux wheel pulls
# ~3 GB of CUDA libraries for nothing.
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# ----- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
# Portable backend by default (MLX cannot run in a Linux container).
ENV HARDWARE_TARGET=other \
    LLM_PROVIDER=ollama \
    EMBED_PROVIDER=ollama \
    TORCH_DEVICE=cpu \
    CHROMA_MODE=http \
    LOG_FORMAT=json

# Copy application code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/eval/ ./data/eval/
COPY pyproject.toml ./

# Non-root user for runtime safety (uid 1000, matches runAsUser in deploy/k8s)
RUN useradd --create-home --uid 1000 --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/')" || exit 1

# Default to the API. UI: `streamlit run src/ui/streamlit_app.py --server.port 8501`
# (with PYTHONPATH=/app). Ingestion: `python -m src.ingestion.pipeline`.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

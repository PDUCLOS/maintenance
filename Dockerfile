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
#   - Future deployment scenarios (cloud GPU/non-MLX backends)
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
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# ----- Stage 2: runtime -----------------------------------------------------
FROM python:3.12-slim AS runtime

WORKDIR /app

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Copy application code
COPY src/ ./src/
COPY pyproject.toml ./

# Non-root user for runtime safety
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8000

# Default to the API. Override with `docker run ... uvicorn src.ui.streamlit_app:app`
# for the UI if you really want to (Streamlit is better run on the host).
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

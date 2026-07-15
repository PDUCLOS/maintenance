#!/usr/bin/env bash
# 02_ingest.sh — Run the full ingestion pipeline.
#
# Reads NASA CMAPSS + PDFs from data/raw/, chunks, embeds, and upserts into
# ChromaDB. Requires:
#   - make chroma-up
#   - make pull-models
#   - data/raw/cmapss/ populated (run 'make data')

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
    echo ">> No venv found. Run 'make setup' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

# Sanity checks
if [[ ! -d data/raw/cmapss ]]; then
    echo ">> ERROR: data/raw/cmapss/ missing. Run 'make data' first." >&2
    exit 1
fi

if ! curl -fsS http://localhost:8001/api/v1/heartbeat >/dev/null 2>&1; then
    echo ">> ERROR: ChromaDB not reachable on :8001. Run 'make chroma-up' first." >&2
    exit 1
fi

echo ">> Running ingestion pipeline..."
python -m src.ingestion.pipeline
echo
echo ">> Done. Start the API with 'make api' and the UI with 'make ui'."

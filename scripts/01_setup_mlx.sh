#!/usr/bin/env bash
# 01_setup_mlx.sh — Download Mistral 7B (MLX) + bge-small embeddings.
#
# One-time download of ~5 GB into the HuggingFace cache.
# Apple Silicon only. Run this once after `make setup`.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ "$(uname -s)" != "Darwin" ]] || [[ "$(uname -m)" != "arm64" ]]; then
    echo "ERROR: MLX requires Apple Silicon (M1/M2/M3/M4/M5)." >&2
    echo "       Detected: $(uname -s) / $(uname -m)" >&2
    exit 1
fi

# Load .env so we read MLX_MODEL_REPO and MLX_EMBED_REPO from one place.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

LLM_REPO="${MLX_MODEL_REPO:-mlx-community/Mistral-7B-Instruct-v0.3-4bit}"
EMBED_REPO="${MLX_EMBED_REPO:-mlx-community/bge-small-en-v1.5-4bit}"

echo ">> Downloading LLM:    $LLM_REPO"
echo ">> Downloading embed:  $EMBED_REPO"
echo "   Cache: $HOME/.cache/huggingface/hub"
echo "   This is a one-time ~5 GB download. Be patient."
echo

if [[ ! -d .venv ]]; then
    echo ">> No venv found. Run 'make setup' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

python - <<PY
from huggingface_hub import snapshot_download
import os

for repo in ["$LLM_REPO", "$EMBED_REPO"]:
    print(f"   Pulling {repo}...")
    snapshot_download(repo)
print("Done.")
PY

echo
echo ">> Models are ready. Next: 'make data' (download NASA CMAPSS), then 'make ingest'."

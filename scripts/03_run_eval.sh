#!/usr/bin/env bash
# 03_run_eval.sh — Run the RAGAS evaluation and snapshot the metrics.

set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -d .venv ]]; then
    echo ">> No venv found. Run 'make setup' first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source .venv/bin/activate

if [[ ! -f data/processed/eval_dataset.jsonl ]]; then
    echo ">> Eval dataset missing. Generating it (this needs CMAPSS data)..."
    python -m src.eval.dataset
fi

echo ">> Running RAGAS evaluation..."
python -m src.eval.ragas_runner
echo
echo ">> Latest snapshot: $(ls -t reports/eval_*.json 2>/dev/null | head -1 || echo 'none')"

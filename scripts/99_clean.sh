#!/usr/bin/env bash
# 99_clean.sh — Full reset: ChromaDB volume, ingested chunks, logs.
# Does NOT remove the HuggingFace model cache (use 'make nuke' for that).

set -euo pipefail

cd "$(dirname "$0")/.."

echo ">> This will DELETE:"
echo "   - the ChromaDB vector store volume"
echo "   - data/processed/chunks.jsonl"
echo "   - data/processed/eval_dataset.jsonl"
echo "   - logs/"
echo
read -p "   Continue? [y/N] " r
[[ "$r" == "y" ]] || { echo "   Aborted."; exit 1; }

echo ">> Stopping ChromaDB..."
docker compose down
echo ">> Removing ChromaDB volume..."
docker volume rm ikc-chroma-data 2>/dev/null || true

echo ">> Removing processed data..."
rm -f data/processed/chunks.jsonl
rm -f data/processed/eval_dataset.jsonl

echo ">> Removing logs..."
rm -rf logs/

echo ">> Done. Next: 'make chroma-up && make ingest' to start fresh."

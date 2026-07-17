"""Quick eval that runs ONLY the per-source retrieval precision metric.

The full RAGAS suite (faithfulness, answer_relevancy, context_precision,
context_recall) needs ~100 LLM calls (~2-3h on M5 Pro because each LLM
call is ~30s). The per-source metric is pure: it only inspects which
chunks the retriever surfaced vs. what the dataset expected.

This script is a fast way to get the per-source number when you don't
have time to wait for the full RAGAS suite. It still runs the same 25
RAG queries (so it still loads the LLM, embeds queries, retrieves) —
that's the unavoidable cost. The skip is just the 4 RAGAS metrics.

Usage:
    .venv/bin/python -m scripts.eval_per_source_only

Snapshot is written to reports/eval_<UTC>.json with the same shape as
a full RAGAS run, but with ONLY per-source metrics. The streamlit UI
will render the per-source column without a row for the standard
RAGAS metrics (gracefully empty).
"""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from src.config import PROJECT_ROOT, settings
from src.eval.ragas_runner import (
    MetricResult,
    _build_ragas_samples,
    _load_dataset,
    _per_source_retrieval_precision,
)
from src.rag.chain import RAGChain
from src.utils.logger import logger


def main() -> Path:
    """Run the per-source eval and write a snapshot. Returns its path."""
    t0 = time.perf_counter()
    items = _load_dataset(settings.eval_dataset_file)
    logger.info("Loaded {} eval items", len(items))

    chain = RAGChain.get()
    samples = _build_ragas_samples(items, chain)

    per_source = _per_source_retrieval_precision(items, samples)
    metric_results = [
        MetricResult(name=f"retrieval@{name}", score=score)
        for name, score in sorted(per_source.items())
    ]
    for m in metric_results:
        logger.info("Per-source retrieval: {} = {:.3f}", m.name, m.score)

    # Snapshot — same shape as the full RAGAS eval so the Streamlit
    # UI can render it without code changes.
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
    out = PROJECT_ROOT / "reports" / f"eval_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": ts,
        "n_samples": len(samples),
        "metrics": [{"name": m.name, "score": m.score} for m in metric_results],
        "eval_kind": "per_source_only",  # marker for the UI
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    elapsed = time.perf_counter() - t0
    logger.info(
        "Per-source eval done in {:.1f}s. Snapshot: {}",
        elapsed,
        out,
    )
    return out


if __name__ == "__main__":
    main()

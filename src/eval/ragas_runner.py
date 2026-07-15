"""Run RAGAS evaluation on the configured dataset.

The runner:
  1. Loads the eval dataset (JSONL: question, ground_truth)
  2. Runs the RAG chain on every question
  3. Feeds (question, answer, contexts, ground_truth) into RAGAS
  4. Snapshots the metrics to reports/eval_<UTC-timestamp>.json

Usage:
    python -m src.eval.ragas_runner
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config import settings
from src.rag.chain import RAGChain
from src.utils.logger import logger
from src.utils.timing import timed


@dataclass
class MetricResult:
    name: str
    score: float


@dataclass
class EvalSnapshot:
    metrics: list[MetricResult]
    n_samples: int
    snapshot_path: Path


@timed
def run(dataset_path: Path, metrics: list[str]) -> EvalSnapshot:
    """Execute the RAGAS evaluation. Returns the snapshot."""
    settings.assert_apple_silicon()
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}. Run 'make eval-dataset' first."
        )

    items = _load_dataset(dataset_path)
    logger.info("Loaded {} eval items from {}", len(items), dataset_path)

    chain = RAGChain.get()
    ragas_samples = []
    for item in items:
        response = chain.query(item["question"])
        ragas_samples.append({
            "question": item["question"],
            "answer": response.answer,
            "contexts": [c.text for c in response.sources],
            "ground_truth": item["ground_truth"],
        })

    # Compute RAGAS metrics
    raise NotImplementedError(
        "RAGAS runner: to be implemented in W3. Use ragas.evaluate(...) with "
        "the chosen metrics, convert to MetricResult list, then snapshot."
    )


def _load_dataset(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _snapshot(metrics: list[MetricResult], n_samples: int) -> EvalSnapshot:
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    out = settings.processed_data_dir.parent / "reports" / f"eval_{ts}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "timestamp_utc": ts,
        "n_samples": n_samples,
        "metrics": [{"name": m.name, "score": m.score} for m in metrics],
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info("Snapshot written to {}", out)
    return EvalSnapshot(metrics=metrics, n_samples=n_samples, snapshot_path=out)


if __name__ == "__main__":
    run(settings.eval_dataset_file, ["faithfulness", "answer_relevancy", "context_precision", "context_recall"])

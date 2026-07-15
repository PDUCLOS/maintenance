"""Integration tests for the RAGAS evaluation pipeline.

Requires:
  - A populated eval_dataset.jsonl (run `make eval-dataset`)
  - Ingested Chroma data and a working RAG chain
"""

from __future__ import annotations

import json
import pytest


@pytest.mark.integration
def test_eval_dataset_minimum_size():
    """The eval dataset should have at least 20 items (PLAN §6)."""
    from src.config import settings

    assert settings.eval_dataset_file.is_file(), (
        f"Eval dataset missing: {settings.eval_dataset_file}. Run 'make eval-dataset'."
    )
    with settings.eval_dataset_file.open() as f:
        items = [json.loads(line) for line in f if line.strip()]
    assert len(items) >= 20


@pytest.mark.integration
def test_eval_metrics_above_threshold():
    """Faithfulness and answer_relevancy should both be > 0.7 (PLAN §6)."""
    from pathlib import Path

    from src.eval.ragas_runner import run as run_ragas

    snapshot = run_ragas(
        dataset_path=Path("data/processed/eval_dataset.jsonl"),
        metrics=["faithfulness", "answer_relevancy"],
    )
    by_name = {m.name: m.score for m in snapshot.metrics}
    assert by_name["faithfulness"] > 0.7, f"faithfulness too low: {by_name['faithfulness']:.2f}"
    assert by_name["answer_relevancy"] > 0.7, f"answer_relevancy too low: {by_name['answer_relevancy']:.2f}"

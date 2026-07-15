"""/eval endpoint — run the RAGAS evaluation suite and snapshot results."""

from __future__ import annotations

import time
from pathlib import Path

from fastapi import APIRouter

from src.api.schemas import EvalMetricResult, EvalRequest, EvalResponse
from src.config import settings
from src.eval.ragas_runner import run as run_ragas
from src.utils.logger import logger

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("", response_model=EvalResponse)
def evaluate(req: EvalRequest) -> EvalResponse:
    """Run RAGAS on the configured evaluation dataset.

    Snapshots the metrics to reports/eval_<timestamp>.json.
    """
    t0 = time.perf_counter()
    dataset_path = Path(req.dataset_path) if req.dataset_path else settings.eval_dataset_file
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Evaluation dataset not found: {dataset_path}. Run 'make eval-dataset' first."
        )
    snapshot = run_ragas(dataset_path=dataset_path, metrics=req.metrics)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("/eval ok (n_samples={}, latency_ms={:.1f})", snapshot.n_samples, elapsed_ms)
    return EvalResponse(
        metrics=[EvalMetricResult(name=m.name, score=m.score) for m in snapshot.metrics],
        n_samples=snapshot.n_samples,
        duration_ms=elapsed_ms,
        snapshot_path=snapshot.snapshot_path,
    )

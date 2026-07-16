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
from datetime import UTC, datetime
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


_METRIC_REGISTRY: dict[str, str] = {
    "faithfulness": "ragas.metrics.faithfulness",
    "answer_relevancy": "ragas.metrics.answer_relevancy",
    "context_precision": "ragas.metrics.context_precision",
    "context_recall": "ragas.metrics.context_recall",
}


def _load_metric(name: str):
    """Import a RAGAS metric by short name. Raises if unknown."""
    if name not in _METRIC_REGISTRY:
        raise ValueError(f"Unknown metric: {name!r}. Known: {list(_METRIC_REGISTRY)}")
    import importlib

    mod_path, attr = _METRIC_REGISTRY[name].rsplit(".", 1)
    mod = importlib.import_module(mod_path)
    return getattr(mod, attr)


def _load_dataset(path: Path) -> list[dict]:
    items: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _build_ragas_samples(items: list[dict], chain: RAGChain) -> list[dict]:
    """Run the RAG chain on every eval question and collect (q, a, ctxs, gt)."""
    samples: list[dict] = []
    for i, item in enumerate(items, start=1):
        question = item["question"]
        try:
            response = chain.query(question)
            answer = response.answer
            contexts = [c.text for c in response.sources]
        except Exception as e:
            logger.warning("Sample {} failed: {}", i, e)
            answer = "ERROR: chain failed"
            contexts = []
        samples.append(
            {
                "question": question,
                "answer": answer,
                "contexts": contexts,
                "ground_truth": item["ground_truth"],
            }
        )
        logger.info("Eval sample {}/{} done", i, len(items))
    return samples


def _build_hf_dataset(samples: list[dict]):
    """Convert samples to a HuggingFace Dataset (what RAGAS expects)."""
    from datasets import Dataset

    return Dataset.from_list(samples)


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
    samples = _build_ragas_samples(items, chain)
    hf_ds = _build_hf_dataset(samples)

    # Resolve metric objects
    metric_objs = [_load_metric(name) for name in metrics]

    # Run RAGAS with our local LLM + embeddings, not RAGAS's OpenAI default
    # — required to keep the "100% local, no API key" claim true.
    from ragas import evaluate
    from ragas.run_config import RunConfig

    from src.rag.embeddings import LangChainEmbedder

    # RAGAS defaults to max_workers=16 (concurrent threads). MLX's Metal
    # command stream is bound to the thread that first touched the GPU —
    # calling our local LLM from 16 worker threads at once makes every
    # thread but one fail with `RuntimeError: There is no Stream(gpu, 0)
    # in current thread.`. We have one local GPU, so force sequential
    # evaluation (max_workers=1) — slower, but the only way every sample
    # actually gets a real score instead of erroring out.
    #
    # timeout=600 (RAGAS default is 180s): a generous timeout matters more
    # here than it would for an API-backed LLM. If a single slow-but-fine
    # MLX generation gets cancelled by asyncio.wait_for, the underlying
    # Python thread keeps running anyway (a ThreadPoolExecutor future can't
    # be interrupted mid-call) — that zombie generation then interleaves
    # with the next job queued on the same dedicated thread and corrupts
    # its output (observed: 1024 repeated "!" characters). A short timeout
    # doesn't skip a slow sample cleanly, it corrupts the *next* one too.
    logger.info("Running RAGAS with metrics: {} (local LLM + embeddings, sequential)", metrics)
    result = evaluate(
        hf_ds,
        metrics=metric_objs,
        llm=chain.llm,
        embeddings=LangChainEmbedder(chain.embedder),
        run_config=RunConfig(max_workers=1, timeout=600),
    )

    # Extract scores (result is a Result object; .scores is a pandas DataFrame)
    metric_results: list[MetricResult] = []
    try:
        scores_df = result.to_pandas()  # one column per metric
    except Exception as e:
        logger.error("Could not convert RAGAS result to DataFrame: {}", e)
        raise
    for name in metrics:
        if name in scores_df.columns:
            mean = float(scores_df[name].mean())
            metric_results.append(MetricResult(name=name, score=mean))
        else:
            logger.warning("Metric {} not in RAGAS result columns", name)

    snapshot = _snapshot(metric_results, len(samples))
    return snapshot


def _snapshot(metrics: list[MetricResult], n_samples: int) -> EvalSnapshot:
    ts = datetime.now(UTC).strftime("%Y-%m-%dT%H-%M-%S")
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
    default_metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    run(settings.eval_dataset_file, default_metrics)

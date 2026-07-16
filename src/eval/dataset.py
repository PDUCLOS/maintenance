"""Generate the evaluation dataset (Q&A pairs) from CMAPSS.

The dataset is generated from the actual training data so ground-truth
answers are deterministic and reproducible. Categories:

    10 factual         — single-stat questions (mean, max, count)
    10 reasoning       — correlation / trend questions
     5 multi-hop       — cross-sensor / cross-cycle questions
     5 out-of-scope    — "I don't know" traps (questions CMAPSS can't answer)

Output: data/processed/eval_dataset.jsonl

Ground truth is computed from the data itself (not hand-written), so it
stays accurate if we re-run the dataset generation after a CMAPSS update.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import settings
from src.ingestion.cmapss_loader import SUBSETS, assert_cmapss_present, load_train
from src.utils.logger import logger
from src.utils.timing import timed

# Out-of-scope questions — the copilot MUST say "I don't know" for these.
OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "What is the price of a new turbofan engine?",
    "Who manufactures the CMAPSS dataset hardware?",
    "Can you predict the exact date of the next engine failure?",
    "What is the phone number of NASA support?",
    "What is the best restaurant in Lyon?",
    "What is the weather forecast for tomorrow in Paris?",
    "Who won the last FIFA World Cup?",
    "What is the capital of Australia?",
]

OUT_OF_SCOPE_ANSWER = "I don't know from the available data."


def _sensor_means_by_subset() -> dict[str, dict[str, float]]:
    """Pre-compute per-subset per-sensor means for faster dataset generation."""
    means: dict[str, dict[str, float]] = {}
    for subset in SUBSETS:
        df = load_train(subset)
        means[subset] = {
            f"sensor_{i:02d}": float(df[f"sensor_{i:02d}"].mean()) for i in range(1, 22)
        }
    return means


def _sensor_trends_by_subset() -> dict[str, dict[str, str]]:
    """Pre-compute per-subset per-sensor trend labels.

    Same definition as `src.ingestion.pipeline._dataframe_to_text`:
    compare first 30% vs last 30% of max cycle.
    """
    trends: dict[str, dict[str, str]] = {}
    for subset in SUBSETS:
        df = load_train(subset)
        max_cycle = int(df["time_cycles"].max())
        cutoff = int(max_cycle * 0.3)
        first_mask = df["time_cycles"] <= cutoff
        last_mask = df["time_cycles"] > (max_cycle - cutoff)
        sub_trends: dict[str, str] = {}
        for i in range(1, 22):
            col = f"sensor_{i:02d}"
            first = df.loc[first_mask, col].mean()
            last = df.loc[last_mask, col].mean()
            if abs(last - first) < 1e-6:
                sub_trends[col] = "stable"
            elif last > first:
                sub_trends[col] = "increases"
            else:
                sub_trends[col] = "decreases"
        trends[subset] = sub_trends
    return trends


def _make_factual(rng: random.Random, means: dict[str, dict[str, float]]) -> list[dict]:
    """10 factual questions: counts, means, max cycles per subset."""
    out: list[dict] = []
    for subset in SUBSETS:
        df = load_train(subset)
        n_units = int(df["unit_nr"].nunique())
        max_cycles = int(df["time_cycles"].max())
        # 3 per subset = 12 total, then we trim to keep the category at 10
        out.append(
            {
                "question": f"How many turbofan engines are in the {subset} training set?",
                "ground_truth": f"{n_units} engines.",
                "category": "factual",
            }
        )
        # pick a random sensor for the mean
        sensor_id = rng.randint(1, 21)
        col = f"sensor_{sensor_id:02d}"
        m = means[subset][col]
        out.append(
            {
                "question": f"What is the mean of {col} across all cycles in {subset}?",
                "ground_truth": f"{m:.2f}.",
                "category": "factual",
            }
        )
        out.append(
            {
                "question": f"What is the maximum number of cycles observed for any unit in {subset}?",
                "ground_truth": f"{max_cycles} cycles.",
                "category": "factual",
            }
        )
    # Trim to 10
    return out[:10]


def _make_reasoning(rng: random.Random, trends: dict[str, dict[str, str]]) -> list[dict]:
    """10 reasoning questions: trend per sensor per subset."""
    out: list[dict] = []
    for subset in SUBSETS:
        # 3 per subset = 12, trim to 10
        for _ in range(3):
            sensor_id = rng.randint(1, 21)
            col = f"sensor_{sensor_id:02d}"
            trend = trends[subset][col]
            out.append(
                {
                    "question": (
                        f"Does {col} tend to increase, decrease, or stay stable as the engine "
                        f"degrades in {subset}?"
                    ),
                    "ground_truth": f"{trend}.",
                    "category": "reasoning",
                }
            )
    return out[:10]


def _make_multi_hop(rng: random.Random) -> list[dict]:
    """5 multi-hop questions: mean of one sensor at an exact cycle."""
    out: list[dict] = []
    for subset in SUBSETS:
        df = load_train(subset)
        # 2 per subset = 8, trim to 5
        for _ in range(2):
            sensor_id = rng.randint(1, 21)
            col = f"sensor_{sensor_id:02d}"
            cycle = rng.randint(50, 200)
            sub = df[df["time_cycles"] == cycle]
            if sub.empty:
                gt = f"No data in {subset} at cycle {cycle}."
            else:
                gt = f"{sub[col].mean():.2f}."
            out.append(
                {
                    "question": f"For {subset}, at cycle {cycle}, what is the mean of {col}?",
                    "ground_truth": gt,
                    "category": "multi_hop",
                }
            )
    return out[:5]


def _make_out_of_scope() -> list[dict]:
    """5 out-of-scope questions: copilot must say 'I don't know'."""
    # Use the first 5 of OUT_OF_SCOPE_QUESTIONS
    return [
        {
            "question": q,
            "ground_truth": OUT_OF_SCOPE_ANSWER,
            "category": "out_of_scope",
        }
        for q in OUT_OF_SCOPE_QUESTIONS[:5]
    ]


@timed
def build() -> Path:
    """Build the evaluation dataset and write it to disk.

    Returns the path to the generated file.
    """
    settings.assert_apple_silicon()
    assert_cmapss_present()
    out_path = settings.eval_dataset_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rng = random.Random(42)  # deterministic
    logger.info("Pre-computing per-sensor stats from CMAPSS data...")
    means = _sensor_means_by_subset()
    trends = _sensor_trends_by_subset()

    items: list[dict] = []
    items.extend(_make_factual(rng, means))
    items.extend(_make_reasoning(rng, trends))
    items.extend(_make_multi_hop(rng))
    items.extend(_make_out_of_scope())

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    n_by_cat: dict[str, int] = {}
    for it in items:
        n_by_cat[it["category"]] = n_by_cat.get(it["category"], 0) + 1
    logger.info(
        "Wrote {} eval items to {} (by category: {})",
        len(items),
        out_path,
        n_by_cat,
    )
    return out_path


if __name__ == "__main__":
    build()

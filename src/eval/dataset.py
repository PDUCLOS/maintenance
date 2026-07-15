"""Generate the evaluation dataset (Q&A pairs) from CMAPSS.

The dataset is generated from the actual training data so ground-truth
answers are deterministic and reproducible. Categories:

    10 factual         — single-stat questions (mean, max, count)
    10 reasoning       — correlation / trend questions
     5 multi-hop       — cross-sensor / cross-cycle questions
     5 out-of-scope    — "I don't know" traps (questions CMAPSS can't answer)

Output: data/processed/eval_dataset.jsonl
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import settings
from src.ingestion.cmapss_loader import SUBSETS, assert_cmapss_present, load_train
from src.utils.logger import logger
from src.utils.timing import timed


# Templates by category. Question + ground_truth are both string templates
# that the dataset generator fills in with real numbers from the data.
FACTUAL_TEMPLATES: list[tuple[str, str]] = [
    (
        "How many turbofan engines are in the {subset} training set?",
        "{n_units} engines.",
    ),
    (
        "What is the mean of sensor_{sensor_id} across all cycles in {subset}?",
        "{mean:.2f}.",
    ),
    (
        "What is the maximum number of cycles observed for any unit in {subset}?",
        "{max_cycles} cycles.",
    ),
]

REASONING_TEMPLATES: list[tuple[str, str]] = [
    (
        "Does sensor_{sensor_id} tend to increase or decrease as the engine degrades in {subset}?",
        "{trend}.",
    ),
]

MULTI_HOP_TEMPLATES: list[tuple[str, str]] = [
    (
        "For {subset}, at cycle {cycle}, what is the mean of sensor_{sensor_id}?",
        "{value:.2f}.",
    ),
]

OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "What is the price of a new turbofan engine?",
    "Who manufactures the CMAPSS dataset hardware?",
    "Can you predict the exact date of the next engine failure?",
    "What is the phone number of NASA support?",
    "What is the best restaurant in Lyon?",
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
    items: list[dict] = []

    for subset in SUBSETS:
        df = load_train(subset)
        # 10 factual
        for _ in range(10):
            tpl_q, tpl_a = rng.choice(FACTUAL_TEMPLATES)
            if "{sensor_id}" in tpl_q:
                sensor_id = rng.randint(1, 21)
                col = f"sensor_{sensor_id:02d}"
                value = float(df[col].mean())
                items.append({
                    "question": tpl_q.format(subset=subset, sensor_id=sensor_id),
                    "ground_truth": tpl_a.format(subset=subset, mean=value, n_units=df["unit_nr"].nunique(), max_cycles=int(df["time_cycles"].max())),
                })
            elif "{n_units}" in tpl_a:
                items.append({
                    "question": tpl_q.format(subset=subset),
                    "ground_truth": tpl_a.format(n_units=df["unit_nr"].nunique()),
                })
            else:
                items.append({
                    "question": tpl_q.format(subset=subset),
                    "ground_truth": tpl_a.format(max_cycles=int(df["time_cycles"].max())),
                })
        # 10 reasoning
        for _ in range(10):
            tpl_q, tpl_a = rng.choice(REASONING_TEMPLATES)
            sensor_id = rng.randint(1, 21)
            col = f"sensor_{sensor_id:02d}"
            # crude trend: compare mean of first 30% vs last 30% of cycles
            cutoff = int(df["time_cycles"].max() * 0.3)
            first = df[df["time_cycles"] <= cutoff][col].mean()
            last = df[df["time_cycles"] > (df["time_cycles"].max() - cutoff)][col].mean()
            trend = "increases" if last > first else "decreases"
            items.append({
                "question": tpl_q.format(subset=subset, sensor_id=sensor_id),
                "ground_truth": tpl_a.format(trend=trend),
            })
        # 5 multi-hop
        for _ in range(5):
            tpl_q, tpl_a = rng.choice(MULTI_HOP_TEMPLATES)
            sensor_id = rng.randint(1, 21)
            col = f"sensor_{sensor_id:02d}"
            cycle = rng.randint(50, 200)
            value = float(df[df["time_cycles"] == cycle][col].mean()) if (df["time_cycles"] == cycle).any() else float("nan")
            items.append({
                "question": tpl_q.format(subset=subset, cycle=cycle, sensor_id=sensor_id),
                "ground_truth": tpl_a.format(value=value) if not value != value else "no data at that cycle",
            })

    # 5 out-of-scope
    for q in OUT_OF_SCOPE_QUESTIONS:
        items.append({
            "question": q,
            "ground_truth": "I don't know from the available data.",
        })

    with out_path.open("w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    logger.info("Wrote {} eval items to {}", len(items), out_path)
    return out_path


if __name__ == "__main__":
    build()

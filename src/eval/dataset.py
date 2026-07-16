"""Generate the evaluation dataset (Q&A pairs) from the PDF catalogue.

The dataset is generated from the actual PDF pages so ground-truth
answers are deterministic and reproducible. Categories:

    10 factual         — single-stat questions on a specific page
     5 reasoning       — "why" / "how" questions answered by the page
     5 retrieval       — exact-citation questions (filename + page)
     5 out-of-scope    — "I don't know" traps

Output: data/processed/eval_dataset.jsonl

The ground truth is the **first non-empty paragraph of a randomly
selected page** (trimmed to <= 280 chars to keep the eval dataset
lean). This is reproducible: pick the same seed, you get the same
questions and the same page-derived answers. The RAGAS judge then
checks whether the LLM's answer is faithful to that paragraph.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

from src.config import settings
from src.ingestion.pdf_loader import load_all_pdfs
from src.utils.logger import logger
from src.utils.timing import timed

# Out-of-scope questions — the copilot MUST say "I don't know" for these.
# These cover common adversarial / off-topic queries: weather, prices,
# restaurants, capitals — the same property (out of scope) is independent
# of the specific corpus (bearing catalogues).
OUT_OF_SCOPE_QUESTIONS: list[str] = [
    "What is the price of a new SKF Explorer bearing?",
    "What is the weather forecast for tomorrow in Paris?",
    "Can you predict the exact date of the next machine failure?",
    "What is the phone number of SKF customer support?",
    "What is the best restaurant in Lyon?",
    "What is the capital of Australia?",
    "Who won the last FIFA World Cup?",
    "What is the current price of Bitcoin?",
]

OUT_OF_SCOPE_ANSWER = "I don't know from the available data."

# Question templates per page. We pick one randomly per page to add
# variety while keeping the Q&A grounded in the actual page content.
QUESTION_TEMPLATES_FACTUAL = [
    "What does this page say about {topic}?",
    "Summarize the key technical specifications on this page.",
    "What is the main topic of this page?",
]

QUESTION_TEMPLATES_REASONING = [
    "Why is {topic} important for bearing maintenance?",
    "What problem does {topic} solve?",
    "When should {topic} be applied?",
]

QUESTION_TEMPLATES_RETRIEVAL = [
    "Which document discusses {topic}?",
    "Where in the catalogue can I find information about {topic}?",
]

# We need a list of plausible topics so the questions read like real
# queries (instead of placeholder "this page" wording). These are
# general bearing-maintenance topics that appear across the catalogues.
TOPICS = [
    "lubrication",
    "load rating",
    "mounting",
    "alignment",
    "vibration",
    "temperature limits",
    "fatigue life",
    "contamination",
    "sealing",
    "bearing clearance",
    "noise diagnosis",
    "grease selection",
    "radial load",
    "axial load",
    "operating temperature",
    "starting torque",
    "static load",
    "dynamic load",
    "service life",
    "failure modes",
]


def _first_meaningful_paragraph(text: str, max_chars: int = 280) -> str:
    """Return the first non-empty paragraph of `text`, trimmed to max_chars.

    Skips lines that are obviously headers / page numbers / TOC entries
    (very short, or all caps, or all digits). Falls back to the first
    200 chars if no good paragraph is found.
    """
    for raw in text.split("\n"):
        line = raw.strip()
        if not line:
            continue
        # Skip headers, footers, page numbers, TOC entries
        if len(line) < 30:
            continue
        if line.isupper():
            continue
        if line.replace(" ", "").isdigit():
            continue
        # Skip lines that are just a section number
        if line.split(".")[0].isdigit() and len(line) < 50:
            continue
        return line[:max_chars]
    return text[:200].strip()


def _make_factual(rng: random.Random, pages: list[tuple[str, str]]) -> list[dict]:
    """10 factual questions: pick 10 random pages, first paragraph = ground truth."""
    out: list[dict] = []
    for source, paragraph in rng.sample(pages, k=min(10, len(pages))):
        topic = rng.choice(TOPICS)
        out.append(
            {
                "question": rng.choice(QUESTION_TEMPLATES_FACTUAL).format(topic=topic),
                "ground_truth": paragraph,
                "category": "factual",
                "expected_source": source,
            }
        )
    return out


def _make_reasoning(rng: random.Random, pages: list[tuple[str, str]]) -> list[dict]:
    """5 reasoning questions."""
    out: list[dict] = []
    for source, paragraph in rng.sample(pages, k=min(5, len(pages))):
        topic = rng.choice(TOPICS)
        out.append(
            {
                "question": rng.choice(QUESTION_TEMPLATES_REASONING).format(topic=topic),
                "ground_truth": paragraph,
                "category": "reasoning",
                "expected_source": source,
            }
        )
    return out


def _make_retrieval(rng: random.Random, pages: list[tuple[str, str]]) -> list[dict]:
    """5 retrieval questions: which document discusses X?"""
    out: list[dict] = []
    for source, _paragraph in rng.sample(pages, k=min(5, len(pages))):
        # extract the filename from the source (source format: "pdf:NAME.pdf:pN")
        filename = source.split(":")[1] if ":" in source else source
        topic = rng.choice(TOPICS)
        out.append(
            {
                "question": rng.choice(QUESTION_TEMPLATES_RETRIEVAL).format(topic=topic),
                "ground_truth": f"{filename} (page {source.split(':p')[-1]}).",
                "category": "retrieval",
                "expected_source": source,
            }
        )
    return out


def _make_out_of_scope() -> list[dict]:
    """5 out-of-scope questions: copilot must say 'I don't know'."""
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
    out_path = settings.eval_dataset_file
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load every PDF page
    pages_raw = load_all_pdfs()
    if not pages_raw:
        raise RuntimeError(
            f"No PDF files found in {settings.pdf_dir}. "
            "Add Schaeffler/SKF catalogues there, then re-run."
        )

    # 2. Extract first meaningful paragraph per page (we keep only pages
    # that yielded a usable paragraph; the rest are skipped)
    pages: list[tuple[str, str]] = []
    for page in pages_raw:
        source = f"pdf:{page.file_name}:p{page.page_number}"
        para = _first_meaningful_paragraph(page.text)
        if para and len(para) >= 30:
            pages.append((source, para))

    if len(pages) < 25:
        raise RuntimeError(
            f"Need at least 25 PDF pages with usable paragraphs, got {len(pages)}. "
            "Add more PDF content or lower the category counts."
        )

    rng = random.Random(42)  # deterministic
    items: list[dict] = []
    items.extend(_make_factual(rng, pages))
    items.extend(_make_reasoning(rng, pages))
    items.extend(_make_retrieval(rng, pages))
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


__all__ = [
    "OUT_OF_SCOPE_ANSWER",
    "OUT_OF_SCOPE_QUESTIONS",
    "TOPICS",
    "build",
]


if __name__ == "__main__":
    build()

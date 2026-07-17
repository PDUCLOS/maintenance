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

Topic ↔ page alignment (commit ecf5643 was the bug fix)
-------------------------------------------------------
Earlier versions picked a random page AND a random topic and stitched
them together without checking that the topic actually appeared in
the page. Result: 13/20 questions had a topic that did NOT match the
expected page (e.g. "Why is sealing important?" expected at a page
about plain bearings). The retriever was correctly returning more
relevant pages from other PDFs, but the dataset's expected_source
was wrong → per-source hit rate was 0% on 4 PDFs.

This version only assigns a question to a page if at least one of
the topics in `TOPICS` actually appears (substring, case-insensitive)
in that page's full text. The topic used in the question is then
picked from the matching subset, so the question ↔ page ↔ topic
triple is always coherent.
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
#
# NOTE: Every factual / reasoning / retrieval template takes a
# `{topic}` placeholder. The previous version had topic-free templates
# ("What is the main topic of this page?") that produced questions
# with zero retrievable signal — the retriever would return whatever
# it wanted, and there was no way to compute a meaningful
# per-source hit rate for those items. They are gone now.
QUESTION_TEMPLATES_FACTUAL = [
    "What does this page say about {topic}?",
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

# Plausible bearing-maintenance topics. These are the things we expect
# a user to ask about; the dataset builder picks a topic that actually
# appears in the page text, so the question is always grounded.
#
# Each entry is (canonical_topic, [synonyms]). Synonyms are matched
# case-insensitively as substrings against the page text. A page is
# considered to "match" a topic if ANY of its synonyms (or the
# canonical name itself) appears in the text.
TOPICS: list[tuple[str, list[str]]] = [
    ("lubrication", ["lubrication", "lubricant", "grease", "oil", "lubrification"]),
    ("load rating", ["load rating", "load ratings", "basic load"]),
    ("mounting", ["mounting", "installation", "fitted"]),
    ("alignment", ["alignment", "aligning", "misalignment"]),
    ("vibration", ["vibration", "vibrations", "vibratory"]),
    ("temperature limits", ["temperature limit", "operating temperature"]),
    ("fatigue life", ["fatigue life", "fatigue", "rating life"]),
    ("contamination", ["contamination", "contaminant", "contaminants", "dirt"]),
    ("sealing", ["sealing", "seal", "seals"]),
    ("bearing clearance", ["bearing clearance", "internal clearance", "clearance"]),
    ("noise diagnosis", ["noise", "acoustic"]),
    ("grease selection", ["grease selection", "grease type", "lubricant selection"]),
    ("radial load", ["radial load", "radial loads"]),
    ("axial load", ["axial load", "axial loads", "thrust load"]),
    ("starting torque", ["starting torque", "breakaway torque"]),
    ("static load", ["static load", "static loads"]),
    ("dynamic load", ["dynamic load", "dynamic loads"]),
    ("service life", ["service life", "lifetime", "operating life"]),
    ("failure modes", ["failure", "failure mode", "failure modes", "damage"]),
    ("axial displacement", ["axial displacement"]),
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


def _page_matching_topics(page_text: str) -> list[str]:
    """Return the canonical names of topics that appear in `page_text`.

    A topic matches if any of its synonyms is found as a substring
    (case-insensitive) of the page text. We pre-lowercase once for
    the whole page so the inner check is a fast `in` on lowercase
    substrings.

    Order is preserved (matches are returned in TOPICS order, not
    document order) so the dataset is deterministic for a given seed.
    """
    lower = page_text.lower()
    matches: list[str] = []
    for canonical, synonyms in TOPICS:
        for syn in synonyms:
            if syn.lower() in lower:
                matches.append(canonical)
                break
    return matches


def _make_factual(
    rng: random.Random,
    pages_with_topics: list[tuple[str, str, list[str]]],
) -> list[dict]:
    """10 factual questions: pick 10 random pages (with at least one
    matching topic), first paragraph = ground truth.
    """
    out: list[dict] = []
    # Sample without replacement, but allow shortfalls (we keep what
    # we can — some pages might be re-used if pool is too small).
    pool = list(pages_with_topics)
    rng.shuffle(pool)
    for source, paragraph, topics in pool[:10]:
        topic = rng.choice(topics)
        out.append(
            {
                "question": rng.choice(QUESTION_TEMPLATES_FACTUAL).format(topic=topic),
                "ground_truth": paragraph,
                "category": "factual",
                "expected_source": source,
            }
        )
    return out


def _make_reasoning(
    rng: random.Random,
    pages_with_topics: list[tuple[str, str, list[str]]],
) -> list[dict]:
    """5 reasoning questions."""
    out: list[dict] = []
    pool = list(pages_with_topics)
    rng.shuffle(pool)
    for source, paragraph, topics in pool[:5]:
        topic = rng.choice(topics)
        out.append(
            {
                "question": rng.choice(QUESTION_TEMPLATES_REASONING).format(topic=topic),
                "ground_truth": paragraph,
                "category": "reasoning",
                "expected_source": source,
            }
        )
    return out


def _make_retrieval(
    rng: random.Random,
    pages_with_topics: list[tuple[str, str, list[str]]],
) -> list[dict]:
    """5 retrieval questions: which document discusses X?"""
    out: list[dict] = []
    pool = list(pages_with_topics)
    rng.shuffle(pool)
    for source, _paragraph, topics in pool[:5]:
        filename = source.split(":")[1] if ":" in source else source
        topic = rng.choice(topics)
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


def _build_pages_with_topics(
    pages_raw: list,  # list[PdfPage]
) -> list[tuple[str, str, list[str]]]:
    """For every PDF page, extract the first meaningful paragraph and
    the list of topics (from `TOPICS`) that appear in the full page text.

    Pages with no matching topic are dropped — they cannot support a
    topic-bearing question without reintroducing the dataset bug we
    just fixed. Pages whose first paragraph is too short (< 30 chars)
    are also dropped (same rule as the old version).
    """
    out: list[tuple[str, str, list[str]]] = []
    dropped_no_topic = 0
    dropped_no_para = 0
    for page in pages_raw:
        para = _first_meaningful_paragraph(page.text)
        if not para or len(para) < 30:
            dropped_no_para += 1
            continue
        topics = _page_matching_topics(page.text)
        if not topics:
            dropped_no_topic += 1
            continue
        source = f"pdf:{page.file_name}:p{page.page_number}"
        out.append((source, para, topics))
    logger.info(
        "Pages: {} usable, {} dropped (no topic match), {} dropped (no paragraph)",
        len(out),
        dropped_no_topic,
        dropped_no_para,
    )
    return out


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

    # 2. Build (source, paragraph, matching_topics) per page.
    # Pages with no matching topic are excluded — they would re-introduce
    # the dataset bug (random topic on a random page).
    pages_with_topics = _build_pages_with_topics(pages_raw)

    # We need at least 20 (10+5+5) pages with topic matches to fill
    # the factual / reasoning / retrieval categories. The corpus has
    # ~5000 pages, so 20 is a tiny fraction. If even one category can't
    # be filled, we want to know loudly rather than silently produce
    # a broken dataset.
    if len(pages_with_topics) < 20:
        raise RuntimeError(
            f"Need at least 20 pages with a topic match, got "
            f"{len(pages_with_topics)}. The TOPICS list may be too narrow "
            f"for the current corpus — extend it and re-run."
        )

    rng = random.Random(42)  # deterministic
    items: list[dict] = []
    items.extend(_make_factual(rng, pages_with_topics))
    items.extend(_make_reasoning(rng, pages_with_topics))
    items.extend(_make_retrieval(rng, pages_with_topics))
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
    "_page_matching_topics",
    "build",
]


if __name__ == "__main__":
    build()

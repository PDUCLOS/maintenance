"""Build a "known page" gold set for retrieval + generation benchmarking.

Why a second dataset? `src/eval/dataset.py` generates questions such as
"What does this page say about alignment?". They are perfect for RAGAS
faithfulness, but useless to measure retrieval: dozens of catalogue pages
talk about alignment, so "did we retrieve page 440?" has no right answer.

Here every question is written FROM one page and is specific enough that
the page is the natural answer (classic known-item / synthetic-query
evaluation):

  1. sample prose-rich pages, stratified by PDF (seed 42, reproducible);
  2. ask a local LLM (gemma4:e4b — a different family from the answering
     model, qwen3.5) for ONE precise technician question + short answer
     drawn only from that page; 1 question in 3 is written in French, to
     exercise cross-lingual retrieval (FR question, EN catalogue);
  3. keep the question, the expected page and the reference answer.

Only questions and page references are written (no page text), so the
file can be committed without redistributing the catalogues.

Usage:
    python scripts/build_gold_set.py [--n 30] [--model gemma4:e4b]
Output: data/eval/gold_known_page.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import settings  # noqa: E402
from src.ingestion.pdf_loader import load_all_pdfs  # noqa: E402

OUT = PROJECT_ROOT / "data" / "eval" / "gold_known_page.jsonl"

PROMPT = """You write evaluation questions for a maintenance copilot that searches \
rolling-bearing catalogues. Below is ONE catalogue page.

Write ONE question that a maintenance technician could ask and that THIS page \
answers precisely. Rules:
- use the specific technical terms of the page (product family, designation, \
property, unit) so the question is not generic;
- never say "this page", "this document" or a page number;
- the answer must be a short fact (one or two sentences) found on the page;
- write the question in {language}.
Return JSON only: {{"question": "...", "answer": "..."}}

[PAGE]
{page}
[/PAGE]"""


def prose_ratio(text: str) -> float:
    """Share of alphabetic characters: filters out pure dimension tables."""
    return sum(ch.isalpha() for ch in text) / max(len(text), 1)


def pick_pages(n: int, seed: int) -> list:
    pages = [p for p in load_all_pdfs() if len(p.text) >= 1200 and prose_ratio(p.text) >= 0.6]
    rng = random.Random(seed)
    by_pdf: dict[str, list] = {}
    for p in pages:
        by_pdf.setdefault(p.file_name, []).append(p)
    for lst in by_pdf.values():
        rng.shuffle(lst)
    picked, i = [], 0
    names = sorted(by_pdf)
    while len(picked) < n and any(by_pdf.values()):
        lst = by_pdf[names[i % len(names)]]
        if lst:
            picked.append(lst.pop())
        i += 1
    return picked


def ask(client: httpx.Client, model: str, page: str, language: str) -> dict | None:
    resp = client.post(
        "/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "user", "content": PROMPT.format(page=page[:3500], language=language)}
            ],
            "stream": False,
            "think": False,
            "format": "json",
            "options": {"temperature": 0.2, "seed": 42, "num_ctx": 4096},
        },
    )
    resp.raise_for_status()
    try:
        data = json.loads(resp.json()["message"]["content"])
    except (json.JSONDecodeError, KeyError):
        return None
    q = str(data.get("question", "")).strip()
    if len(q) < 15 or re.search(r"\b(this page|this document|cette page|ce document)\b", q, re.I):
        return None
    return {"question": q, "answer": str(data.get("answer", "")).strip()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--model", default="gemma4:e4b")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pages = pick_pages(args.n + 10, args.seed)  # spare pages if the LLM output is unusable
    items: list[dict] = []
    with httpx.Client(base_url=settings.ollama_base_url, timeout=180) as client:
        for page in pages:
            if len(items) >= args.n:
                break
            language = "French" if len(items) % 3 == 2 else "English"
            qa = ask(client, args.model, page.text, language)
            if qa is None:
                print(f"  skip {page.file_name} p{page.page_number} (unusable LLM output)")
                continue
            items.append(
                {
                    "id": f"kp{len(items) + 1:02d}",
                    "question": qa["question"],
                    "reference_answer": qa["answer"],
                    "expected_source": f"pdf:{page.file_name}:p{page.page_number}",
                    "language": "fr" if language == "French" else "en",
                    "generator": args.model,
                }
            )
            print(f"  {items[-1]['id']} [{items[-1]['language']}] {qa['question'][:90]}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"{len(items)} questions -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

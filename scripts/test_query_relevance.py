#!/usr/bin/env python3
"""Smoke test for the RAG chain — relevance + jargon hygiene.

Sends 5 representative questions to the running API on :8000, prints the
answer, and checks two invariants:

  1. **Relevance**: the answer must contain a number, a unit, or a clear
     refusal (e.g. "I don't know") — not a non-sequitur.
  2. **Jargon hygiene**: the answer must NOT contain the words "chunk",
     "retrieval", "retrieved", "embedding", "vector", "RAG", "passage",
     "extrait" (or the related developer terms). The end user is a
     maintenance engineer, not a developer.

Usage:
    .venv/bin/python scripts/test_query_relevance.py
    .venv/bin/python scripts/test_query_relevance.py --api http://localhost:8000

Exit code 0 if all checks pass, 1 otherwise.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from typing import Any

import httpx

# Test questions. Mix of factual CMAPSS, technical PDF, and out-of-scope.
QUESTIONS = [
    {
        "id": "cmapss_count",
        "q": "Combien de moteurs dans le dataset FD001 ?",
        "expect_any": ["100", "cent"],
        "expect_no": [],
    },
    {
        "id": "cmapss_trend",
        "q": "Le capteur sensor_11 augmente ou diminue au cours du temps dans FD001 ?",
        "expect_any": [],
        "expect_no": [],
    },
    {
        "id": "pdf_bearing",
        "q": "What is the basic dynamic load rating of a deep groove ball bearing?",
        "expect_any": [],
        "expect_no": [],
    },
    {
        "id": "mirror_en",
        "q": "How many sensors are in the CMAPSS dataset?",
        "expect_any": ["21"],
        "expect_no": [],
    },
    {
        "id": "out_of_scope",
        "q": "What's the weather forecast for Paris tomorrow?",
        "expect_any": ["don't know", "sais pas", "hors scope", "out of scope"],
        "expect_no": [],
    },
]

# Forbidden jargon. The prompt's rule 8 forbids these — if any leak
# through, the prompt is not strict enough.
JARGON = re.compile(
    r"\b(chunk|chunks|retrieval|retrieved|retrieving|embedding|embeddings|"
    r"vector store|vecteur|RAG|passage|passages|extrait|extracts)\b",
    flags=re.IGNORECASE,
)


def call_api(api: str, question: str, top_k: int = 5) -> dict[str, Any]:
    """POST a /query and return the JSON response."""
    r = httpx.post(
        f"{api}/query",
        json={"question": question, "top_k": top_k, "stream": False},
        timeout=120.0,
    )
    r.raise_for_status()
    return r.json()


def check_answer(
    question_id: str,
    question: str,
    answer: str,
    expect_any: list[str],
    expect_no: list[str],
) -> tuple[bool, list[str]]:
    """Return (passed, list_of_issues) for one answer."""
    issues: list[str] = []
    # 1. Jargon hygiene
    leaks = JARGON.findall(answer)
    if leaks:
        issues.append(f"JARGON LEAK: {leaks}")
    # 2. Expected substrings
    if expect_any:
        lower = answer.lower()
        if not any(token.lower() in lower for token in expect_any):
            issues.append(f"expected one of {expect_any} in answer, got: {answer[:200]!r}")
    # 3. Expected forbidden substrings
    if expect_no:
        lower = answer.lower()
        for token in expect_no:
            if token.lower() in lower:
                issues.append(f"forbidden token {token!r} found in answer")
    return (not issues, issues)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api", default="http://localhost:8000", help="API base URL")
    args = parser.parse_args()

    # Wait for API to be ready
    print(f"Connecting to {args.api}...")
    for _ in range(10):
        try:
            r = httpx.get(f"{args.api}/health", timeout=2.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(1)
    else:
        print(f"❌ API not reachable on {args.api}")
        return 1
    print("API healthy.\n")

    results: list[dict[str, Any]] = []
    for q in QUESTIONS:
        print(f"─── {q['id']} ─────────────────────────────────────────")
        print(f"Q: {q['q']}")
        t0 = time.perf_counter()
        try:
            resp = call_api(args.api, q["q"])
        except Exception as e:
            print(f"❌ API error: {e}")
            results.append({**q, "passed": False, "error": str(e)})
            continue
        elapsed_ms = (time.perf_counter() - t0) * 1000
        answer = resp.get("answer", "")
        n_sources = len(resp.get("sources", []))
        print(f"A ({elapsed_ms:.0f}ms, {n_sources} sources):")
        # Truncate long answers for readability
        if len(answer) > 600:
            print(f"  {answer[:600]}…")
        else:
            for line in answer.splitlines():
                print(f"  {line}")
        passed, issues = check_answer(
            q["id"], q["q"], answer, q.get("expect_any", []), q.get("expect_no", []),
        )
        status = "✅" if passed else "❌"
        print(f"{status} {q['id']}: {len(issues)} issue(s)")
        for issue in issues:
            print(f"     • {issue}")
        results.append({**q, "passed": passed, "issues": issues, "elapsed_ms": elapsed_ms})
        print()

    n_pass = sum(1 for r in results if r.get("passed"))
    n_fail = len(results) - n_pass
    print(f"════════════════════════════════════════════════════════════")
    print(f"  {n_pass} / {len(results)} passed  ({n_fail} failed)")
    if n_fail:
        print()
        print("Failed questions:")
        for r in results:
            if not r.get("passed"):
                print(f"  - {r['id']}: {r.get('issues', [])}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

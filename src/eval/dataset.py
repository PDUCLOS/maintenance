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
from src.rag.language import detect_language
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
#
# Each category has BOTH an EN and an FR template list. The dataset
# builder picks the right list based on the language of the page
# (detected via `src.rag.language.detect_language`). This matters for
# the FR-only NTN-SNR catalogue + diagnostic guide: the dense retriever
# (bge-m3) is multilingual, but an EN query for a topic ("What is
# the rating life of a bearing?") surfaces EN content (SKF/Schaeffler)
# even when the same topic is better documented in FR (NTN-SNR). A
# FR question surfaces the NTN-SNR content correctly. So the dataset
# is bilingual by design: each page's question is in the page's own
# language, which is what a real user of that document would type.
QUESTION_TEMPLATES_FACTUAL = {
    "en": [
        "What does this page say about {topic}?",
    ],
    "fr": [
        "Que dit cette page à propos de {topic} ?",
        "Que trouve-t-on sur cette page concernant {topic} ?",
    ],
}

# Reasoning templates are GENDER-NEUTRAL on purpose. French adjectives
# and past participles agree with the noun they modify (m./f.), and
# {topic} could be either ("le montage" = m., "la lubrification" = f.).
# Using "est" + infinitive ("est important") sidesteps the agreement
# problem without sounding unnatural. The previous "est-il" template
# produced "Pourquoi lubrification est-il..." which is wrong grammar
# (lubrification is feminine → "est-elle").
# Reasoning templates are GENDER-NEUTRAL on purpose. French verbs
# agree with their subject (m./f., s./pl.), and {topic} is a noun
# phrase whose gender is baked into the article ("le montage" = m.,
# "la lubrification" = f., "les défaillances" = f. pl.). The previous
# templates ("Pourquoi {topic} est important..." / "À quel problème
# {topic} répond-il...") produced ungrammatical outputs like "À quel
# problème les défaillances répond-il" (subject "les défaillances"
# is f. pl., verb form "répond-il" is m. s. — agreement broken).
# The "on" form is impersonal (3rd person s., no agreement) and reads
# naturally in technical French. "Quel problème nécessite {topic} ?"
# uses a 3rd-person-singular verb that doesn't agree with its
# object ("nécessite" doesn't take a m./f. inflection).
QUESTION_TEMPLATES_REASONING = {
    "en": [
        "Why is {topic} important for bearing maintenance?",
        "What problem does {topic} solve?",
        "When should {topic} be applied?",
    ],
    "fr": [
        "Pourquoi s'intéresse-t-on à {topic} dans la maintenance des roulements ?",
        "Quel problème nécessite {topic} ?",
        "Quand faut-il appliquer {topic} ?",
    ],
}

QUESTION_TEMPLATES_RETRIEVAL = {
    "en": [
        "Which document discusses {topic}?",
        "Where in the catalogue can I find information about {topic}?",
    ],
    "fr": [
        # "sur {topic}" (instead of "de {topic}") avoids the
        # "de le" / "du" / "des" elision that depends on the
        # article baked into {topic}.
        "Quel document contient des informations sur {topic} ?",
        "Où trouve-t-on des informations sur {topic} dans le catalogue ?",
    ],
}


def _format_question(
    category: str,  # "factual" | "reasoning" | "retrieval"
    full_page_text: str,
    paragraph: str,
    canonical_topic: str,
    rng: random.Random,
) -> tuple[str, str]:
    """Pick a question template in the page's language, with the topic
    in that same language.

    Returns the rendered question AND the detected page language
    (so the caller can record it in the snapshot if useful).

    We detect the language on the FULL page text, not just the first
    paragraph. Some pages (e.g. the NTN-SNR diagnostic guide) are
    mostly tables with a short FR intro paragraph that starts with
    EN-style bullets — detecting only on the intro would mis-classify
    them as EN and emit an English question for a French page. The
    `detect_language` heuristic is robust to mixed content as long
    as the dominant language has more function words than the other.
    """
    lang = detect_language(full_page_text)  # "fr" or "en"
    templates_by_lang = {
        "factual": QUESTION_TEMPLATES_FACTUAL,
        "reasoning": QUESTION_TEMPLATES_REASONING,
        "retrieval": QUESTION_TEMPLATES_RETRIEVAL,
    }[category]
    templates = templates_by_lang[lang]
    # If the page language is FR, translate the topic to FR (if we
    # have a translation) so the question reads naturally.
    if lang == "fr":
        topic_for_q = TOPIC_FR.get(canonical_topic, canonical_topic)
    else:
        topic_for_q = canonical_topic
    question = rng.choice(templates).format(topic=topic_for_q)
    return question, lang


# FR equivalents of the canonical EN topic names, WITH the article
# (le / la / les / l') so the resulting noun phrase can be slotted
# directly into a FR template without grammatical agreement issues
# downstream. Example: "lubrification" → "la lubrification" so the
# template "à propos de {topic}" becomes "à propos de la lubrification"
# (correct, no "à propos de lubrification" which reads as franglais).
#
# The article is also the gender carrier: the templates below are
# designed to NOT need to agree with it (verb in 3rd-person singular
# is impersonal, adjectives are avoided).
TOPIC_FR: dict[str, str] = {
    "lubrication": "la lubrification",
    "load rating": "la capacité de charge",
    "mounting": "le montage",
    "alignment": "l'alignement",
    "vibration": "les vibrations",
    "temperature limits": "la température",
    "fatigue life": "la durée de vie",
    "contamination": "la pollution",
    "sealing": "l'étanchéité",
    "bearing clearance": "le jeu interne",
    "noise diagnosis": "le bruit",
    "grease selection": "le graissage",
    "radial load": "la charge radiale",
    "axial load": "la charge axiale",
    "starting torque": "le couple de démarrage",
    "static load": "la charge statique",
    "dynamic load": "la charge dynamique",
    "service life": "la durée de service",
    "failure modes": "les défaillances",
    "axial displacement": "le déplacement axial",
    "diagnostic method": "le diagnostic",
}

# Plausible bearing-maintenance topics. These are the things we expect
# a user to ask about; the dataset builder picks a topic that actually
# appears in the page text, so the question is always grounded.
#
# Each entry is (canonical_topic, [synonyms]). Synonyms are matched
# case-insensitively as substrings against the page text. A page is
# considered to "match" a topic if ANY of its synonyms (or the
# canonical name itself) appears in the text.
#
# Bilingual coverage: the corpus is a mix of EN (Schaeffler / SKF
# catalogues) and FR (NTN-SNR catalogue + diagnostic guide). The
# synonyms include both EN and FR technical terms so a French page
# is reachable from a French-style topic match. The topic CANONICAL
# name stays in English (for stable, language-agnostic labels in the
# generated Q&A + snapshot reports), but the matching vocabulary is
# bilingual.
#
# Note on the FR side: I picked longer, more specific terms ("joints",
# "graissage", "durée de vie") over short common words ("joint",
# "jeu", "vie") to avoid spurious matches against EN pages that happen
# to contain the same short string. If a topic needs a short FR
# keyword to match a real document, the long form is included too.
TOPICS: list[tuple[str, list[str]]] = [
    (
        "lubrication",
        [
            "lubrication",
            "lubricant",
            "grease",
            "oil",
            # FR
            "lubrification",
            "graissage",
            "graisse",
            "lubrifiant",
        ],
    ),
    (
        "load rating",
        [
            "load rating",
            "load ratings",
            "basic load",
            # FR
            "capacité de charge",
            "charge de base",
            "charge nominale",
        ],
    ),
    (
        "mounting",
        [
            "mounting",
            "installation",
            "fitted",
            # FR
            "montage",
            "démontage",
            "pose",
            "dépose",
        ],
    ),
    (
        "alignment",
        [
            "alignment",
            "aligning",
            "misalignment",
            # FR
            "alignement",
            "désalignement",
            "défaut d'alignement",
        ],
    ),
    (
        "vibration",
        [
            "vibration",
            "vibrations",
            "vibratory",
            # FR
            "vibration",
            "vibratoire",
            "vibratoires",
        ],
    ),
    (
        "temperature limits",
        [
            "temperature limit",
            "operating temperature",
            # FR
            "température",
            "limite de température",
            "plage de température",
        ],
    ),
    (
        "fatigue life",
        [
            "fatigue life",
            "fatigue",
            "rating life",
            # FR
            "durée de vie",
            "endurance",
            "limite de fatigue",
        ],
    ),
    (
        "contamination",
        [
            "contamination",
            "contaminant",
            "contaminants",
            "dirt",
            # FR
            "pollution",
            "propreté",
            "contaminant",
            "polluant",
        ],
    ),
    (
        "sealing",
        [
            "sealing",
            "seal",
            "seals",
            # FR
            "étanchéité",
            "joint",
            "joints",
            "joint d'étanchéité",
        ],
    ),
    (
        "bearing clearance",
        [
            "bearing clearance",
            "internal clearance",
            "clearance",
            # FR
            "jeu interne",
            "jeu radial",
            "jeu de fonctionnement",
        ],
    ),
    (
        "noise diagnosis",
        [
            "noise",
            "acoustic",
            # FR
            "bruit",
            "acoustique",
            "diagnostic vibratoire",
            "analyse vibratoire",
        ],
    ),
    (
        "grease selection",
        [
            "grease selection",
            "grease type",
            "lubricant selection",
            # FR
            "sélection de graisse",
            "choix de graisse",
            "type de lubrifiant",
        ],
    ),
    (
        "radial load",
        [
            "radial load",
            "radial loads",
            # FR
            "charge radiale",
            "charges radiales",
        ],
    ),
    (
        "axial load",
        [
            "axial load",
            "axial loads",
            "thrust load",
            # FR
            "charge axiale",
            "charges axiales",
            "charge de poussée",
        ],
    ),
    (
        "starting torque",
        [
            "starting torque",
            "breakaway torque",
            # FR
            "couple de démarrage",
            "couple initial",
            "couple de rupture",
        ],
    ),
    (
        "static load",
        [
            "static load",
            "static loads",
            # FR
            "charge statique",
            "charges statiques",
        ],
    ),
    (
        "dynamic load",
        [
            "dynamic load",
            "dynamic loads",
            # FR
            "charge dynamique",
            "charges dynamiques",
        ],
    ),
    (
        "service life",
        [
            "service life",
            "lifetime",
            "operating life",
            # FR
            "durée de service",
            "durée de vie utile",
            "longévité",
        ],
    ),
    (
        "failure modes",
        [
            "failure",
            "failure mode",
            "failure modes",
            "damage",
            # FR
            "défaillance",
            "mode de défaillance",
            "défaillances",
            "endommagement",
            "défaut",
            "panne",
        ],
    ),
    (
        "axial displacement",
        [
            "axial displacement",
            # FR
            "déplacement axial",
            "déplacements axiaux",
        ],
    ),
    # Bonus topic for the NTN-SNR diagnostic guide (a FR-only doc).
    # Standard catalogue topics don't cover its specific vocabulary
    # (cause → symptom chains) so we add an explicit "diagnostic"
    # anchor that maps to both the dataset's failure-modes content
    # and the diagnostic guide's structure.
    (
        "diagnostic method",
        [
            "diagnostic",
            "troubleshooting",
            "root cause",
            # FR
            "diagnostic",
            "dépannage",
            "recherche de cause",
            "arbre de défaillance",
        ],
    ),
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
    pages_with_topics: list[tuple[str, str, str, list[str]]],
) -> list[dict]:
    """10 factual questions: 1 question per major PDF (round-robin
    stratified sampling), first paragraph = ground truth.

    Each question is rendered in the page's own language (EN or FR),
    so the dense retriever's multilingual embedding finds the right
    document regardless of which language the user would type.
    """
    picked = _sample_stratified_by_pdf(rng, pages_with_topics, 10)
    out: list[dict] = []
    for source, paragraph, full_text, topics in picked:
        canonical_topic = rng.choice(topics)
        question, lang = _format_question("factual", full_text, paragraph, canonical_topic, rng)
        out.append(
            {
                "question": question,
                "ground_truth": paragraph,
                "category": "factual",
                "expected_source": source,
                "language": lang,
            }
        )
    return out


def _make_reasoning(
    rng: random.Random,
    pages_with_topics: list[tuple[str, str, str, list[str]]],
) -> list[dict]:
    """5 reasoning questions: stratified by PDF (round-robin)."""
    picked = _sample_stratified_by_pdf(rng, pages_with_topics, 5)
    out: list[dict] = []
    for source, paragraph, full_text, topics in picked:
        canonical_topic = rng.choice(topics)
        question, lang = _format_question("reasoning", full_text, paragraph, canonical_topic, rng)
        out.append(
            {
                "question": question,
                "ground_truth": paragraph,
                "category": "reasoning",
                "expected_source": source,
                "language": lang,
            }
        )
    return out


def _make_retrieval(
    rng: random.Random,
    pages_with_topics: list[tuple[str, str, str, list[str]]],
) -> list[dict]:
    """5 retrieval questions: stratified by PDF (round-robin)."""
    picked = _sample_stratified_by_pdf(rng, pages_with_topics, 5)
    out: list[dict] = []
    for source, paragraph, full_text, topics in picked:
        filename = source.split(":")[1] if ":" in source else source
        canonical_topic = rng.choice(topics)
        question, lang = _format_question("retrieval", full_text, paragraph, canonical_topic, rng)
        out.append(
            {
                "question": question,
                "ground_truth": f"{filename} (page {source.split(':p')[-1]}).",
                "category": "retrieval",
                "expected_source": source,
                "language": lang,
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
) -> list[tuple[str, str, str, list[str]]]:
    """For every PDF page, extract:
      - the source string ("pdf:NAME.pdf:pN")
      - the first meaningful paragraph (for the ground truth + first-
        paragraph checks in tests)
      - the FULL page text (for accurate language detection — short
        paragraphs dominated by tables or bullet lists can be
        mis-classified)
      - the list of topics (from `TOPICS`) that appear in the full
        page text (for the random topic pick)

    Returns a list of 4-tuples. We keep the full page text in memory
    (not just the paragraph) because the dataset needs both: the
    paragraph is the ground truth for the eval, the full text is
    used for language detection + topic matching (which already
    happens here, in this function). Memory cost: ~2-3 KB per page
    times ~4000 pages = ~10 MB total — fine.

    Pages with no matching topic are dropped — they cannot support a
    topic-bearing question without reintroducing the dataset bug we
    just fixed. Pages whose first paragraph is too short (< 30 chars)
    are also dropped (same rule as the old version).
    """
    out: list[tuple[str, str, str, list[str]]] = []
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
        out.append((source, para, page.text, topics))
    logger.info(
        "Pages: {} usable, {} dropped (no topic match), {} dropped (no paragraph)",
        len(out),
        dropped_no_topic,
        dropped_no_para,
    )
    return out


def _sample_stratified_by_pdf(
    rng: random.Random,
    pages: list[tuple[str, str, str, list[str]]],
    n: int,
) -> list[tuple[str, str, str, list[str]]]:
    """Pick `n` pages, ROUND-ROBIN across PDFs.

    Pure random sampling would let the corpus skew decide which PDFs
    are represented — Schaeffler/SKF dominate by chunk count, NTN-SNR
    (the only FR-only document) is ~1.5% of the corpus. With random
    sampling and seed=42, NTN-SNR could easily end up with zero
    questions in the dataset, which defeats the whole point of adding
    French synonyms to `TOPICS`.

    Round-robin guarantees every PDF gets at least 1 pick (if we
    have enough slots), and small PDFs (NTN-SNR, GGB) get the same
    floor as big ones (Schaeffler, SKF). When `n` exceeds the number
    of PDFs, we cycle again with a fresh shuffle per pass so the
    second pick for a big PDF is still random across its pages.

    Parameters
    ----------
    rng : random.Random
        The deterministic RNG (so seed=42 gives the same dataset on
        every run).
    pages : list of (source, paragraph, full_text, topics)
        The candidate pool (already filtered: has a topic match +
        has a usable first paragraph).
    n : int
        How many pages to return. `n > len(pages)` is allowed —
        we wrap around and re-shuffle per PDF (page-level sampling
        is still random within each PDF; the round-robin just
        governs WHICH PDF we draw from).
    """
    if n <= 0 or not pages:
        return []

    # Group by PDF, in encounter order (stable across calls).
    # The inner type is (source, paragraph, full_text, topics) —
    # we don't annotate it in full because nested-bracket parsing in
    # older Python type-checkers (and in ruff's AST scanner) gets
    # confused by `list[tuple[...]]` inside a dict[...].
    by_pdf: dict = {}
    for entry in pages:
        pdf = entry[0].split(":")[1] if ":" in entry[0] else entry[0]
        by_pdf.setdefault(pdf, []).append(entry)
    pdfs = list(by_pdf.keys())
    if not pdfs:
        return []

    # Shuffle within each PDF so the first page isn't always p1.
    # (Seed=42 keeps this deterministic across runs.)
    for pdf in pdfs:
        rng.shuffle(by_pdf[pdf])

    out: list = []
    cursor = 0  # index into `pdfs` for the round-robin
    while len(out) < n:
        pdf = pdfs[cursor % len(pdfs)]
        bucket = by_pdf[pdf]
        # If we've exhausted this PDF, drop it from the cycle for
        # this draw (we'll re-add it on a future call with a new
        # pool if needed). Reshuffle to allow wrap-around.
        if not bucket:
            # Last resort: cycle through the remaining PDFs only
            pdfs = [p for p in pdfs if by_pdf[p]] or pdfs
            if not by_pdf[pdf]:
                continue
            bucket = by_pdf[pdf]
        # Draw one from this PDF. The bucket is reshuffled every full
        # round, so when we come back to this PDF on the next cycle
        # the page we draw is (with high probability) a different one.
        if len(bucket) == 1:
            out.append(bucket[0])
        else:
            # Each PDF's bucket is reshuffled, so popping from the
            # end gives a different page each pass.
            page = bucket.pop()
            out.append(page)
        cursor += 1
        if cursor % len(pdfs) == 0 and len(out) < n:
            # End of a round — re-shuffle every non-empty bucket so
            # the next round picks different pages from the same PDFs.
            for pdf in pdfs:
                if len(by_pdf[pdf]) > 1:
                    rng.shuffle(by_pdf[pdf])
    return out[:n]


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

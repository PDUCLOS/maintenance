"""Language detection and bilingual handling.

The project's corpus is mostly English (Schaeffler + SKF catalogues,
the catalogues) but the user may ask questions in French or
English. The RAG chain must:

1. **Detect the language of the question** automatically.
2. **Answer in the same language** as the question (mirror response).
3. **Keep the retrieved sources in their original language** — they are
   the ground truth, we don't translate them.

The detection is a lightweight heuristic (no extra dependency). It counts
common words from each language and picks the one with the higher score.
For short queries (< 5 words) the score is unreliable — we fall back to
the default ("en") in that case.

This module exposes a single public function:
    detect_language(text: str) -> Literal["fr", "en"]
"""

from __future__ import annotations

# Compact list of very common words per language. We use a small
# vocabulary on purpose: long lists slow down the scan and bias the
# result on technical text. The chosen words are all distinct enough
# to be unambiguous in short queries.
_FR_HINTS: frozenset[str] = frozenset({
    # function words
    "le", "la", "les", "un", "une", "des", "du", "de", "d'un", "d'une",
    "et", "ou", "mais", "donc", "car", "si", "que", "quoi", "quel", "quelle",
    "qui", "comment", "pourquoi", "combien", "où", "ce", "cette", "ces",
    "mon", "ton", "son", "ma", "ta", "sa", "mes", "tes", "ses", "notre",
    "votre", "leur", "dans", "sur", "sous", "avec", "sans", "pour", "par",
    "est", "sont", "était", "étaient", "fait", "faits", "peut", "peuvent",
    "je", "tu", "il", "elle", "on", "nous", "vous", "ils", "elles",
    "ne", "pas", "plus", "aussi", "très", "déjà", "encore", "sais", "sait", "savent",
    # common nouns / adjectives
    "roulement", "roulements", "moteur", "moteurs", "capteur", "capteurs",
    "température", "pression", "vitesse", "durée", "nombre",
    "moyenne", "moyen", "cycle", "cycles", "panne", "pannes", "quels", "quelles",
})

_EN_HINTS: frozenset[str] = frozenset({
    # function words
    "the", "a", "an", "of", "in", "on", "at", "to", "for", "with", "without",
    "and", "or", "but", "so", "because", "if", "that", "what", "which", "who",
    "how", "why", "when", "where", "this", "these", "those",
    "my", "your", "his", "her", "its", "our", "their",
    "is", "are", "was", "were", "been", "be", "has", "have", "had",
    "do", "does", "did", "can", "could", "should", "would", "will",
    # common nouns / adjectives in our domain
    "bearing", "bearings", "engine", "engines", "sensor", "sensors",
    "temperature", "pressure", "speed", "duration", "cycle", "cycles",
    "failure", "failures", "mean", "average", "median", "max", "min",
    "many", "much",
})


def detect_language(text: str) -> str:
    """Detect the language of `text`. Returns "fr" or "en".

    Algorithm: tokenise on whitespace + common punctuation, lowercase,
    count how many tokens appear in each language's hint set. Return
    the language with the higher count; default to "en" on a tie
    (English is the dominant language in our corpus, and the LLM
    handles EN queries reliably).

    Short queries (≤ 5 tokens) are too short for reliable detection.
    We still try — the score will just be 0/0 on most queries, and the
    tiebreak default to "en" applies.
    """
    if not text or not text.strip():
        return "en"

    # Tokenise: split on whitespace and strip common punctuation.
    import re
    tokens = re.findall(r"[a-zA-ZÀ-ÿ]+", text.lower())
    if not tokens:
        return "en"

    fr_score = sum(1 for t in tokens if t in _FR_HINTS)
    en_score = sum(1 for t in tokens if t in _EN_HINTS)

    if fr_score > en_score:
        return "fr"
    return "en"  # default, including ties


__all__ = ["detect_language"]

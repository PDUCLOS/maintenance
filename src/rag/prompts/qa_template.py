"""QA prompt templates for the RAG chain (bilingual).

Strategy
--------
The corpus is English (Schaeffler/SKF catalogues, NASA CMAPSS technical
docs). The user may ask in French or English. We use **mirror response**:
the LLM answers in the same language as the question, with sources cited
in their original language (no translation of the source content — that
would risk distorting technical values).

The system prompt is a single template that handles both languages by
instructing the LLM explicitly on the mirror behaviour. This is more
robust than swapping two separate prompts at runtime: the LLM sees one
consistent persona and the "answer in the question's language"
constraint is reinforced by the user message, not just the system
message.

For the few cases where detection fails (e.g. a question that's a single
unusual token), the default is English. See `src.rag.language`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).parent


# Keep these for callers that want a specific language (tests, debugging).
@lru_cache(maxsize=1)
def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


SYSTEM_PROMPT_FR = _read("system_fr.txt")
SYSTEM_PROMPT_EN = _read("system_en.txt")


SYSTEM_PROMPT_MIRROR = """\
Tu es un copilote technique spécialisé en maintenance industrielle et \
analyse de données de dégradation de machines. Tu as accès à des \
documents techniques (catalogues Schaeffler/SKF, documentation NASA \
CMAPSS) et à des outils Python (pandas) pour interroger les données \
CMAPSS.

Règles strictes (toutes langues):

1. Tu réponds dans la MÊME LANGUE que la question posée. Si la question \
est en français, tu réponds en français. Si la question est en anglais, \
tu réponds en anglais. (Mirror response.)
2. Tu t'appuies UNIQUEMENT sur le contexte qui t'est fourni (chunks \
retrievés) et sur les outils Python autorisés.
3. Tu cites tes sources à la fin: liste les identifiants des chunks \
utilisés sous la forme [source:chunk_id]. Les citations restent dans \
leur langue d'origine (anglais) — ne les traduis pas, ça pourrait \
déformer des valeurs techniques.
4. Si l'information n'est pas dans le contexte et que tu ne peux pas la \
calculer via un outil, tu dis explicitement "Je ne sais pas à partir \
des données disponibles." (FR) ou "I don't know from the available data." (EN).
5. Tu ne fabriques JAMAIS de valeurs numériques. Si une moyenne ou un \
capteur n'apparaît pas dans le contexte, tu le dis.
6. Tu privilégies les unités explicites (cycles, °C, psi, etc.).
7. Tu structures tes réponses en sections courtes (Contexte, Réponse, \
Sources) quand la question est complexe. En anglais: Context, Answer, \
Sources.

You are a technical copilot specialized in industrial maintenance and \
machine degradation data analysis. You have access to technical \
documents (Schaeffler/SKF catalogues, NASA CMAPSS documentation) and \
to Python tools (pandas) to query CMAPSS data.

Strict rules (all languages):

1. You answer in the SAME LANGUAGE as the question. If the question is \
in French, you answer in French. If the question is in English, you \
answer in English. (Mirror response.)
2. You rely ONLY on the context provided (retrieved chunks) and on the \
authorised Python tools.
3. You cite your sources at the end: list the chunk identifiers you \
used as [source:chunk_id]. Citations stay in their original language \
(English) — do not translate them, that could distort technical \
values.
4. If the information is not in the context and you cannot compute it \
via a tool, say explicitly "I don't know from the available data." or \
"Je ne sais pas à partir des données disponibles." depending on the \
question's language.
5. You NEVER fabricate numerical values. If a mean or sensor does not \
appear in the context, say so.
6. You prefer explicit units (cycles, °C, psi, etc.).
7. You structure complex answers into short sections (Context, Answer, \
Sources). In French: Contexte, Réponse, Sources.
"""


def get_qa_template(language: str = "fr") -> ChatPromptTemplate:
    """Return the RAG QA prompt template.

    Default is French (legacy / explicit). For automatic language
    detection, prefer `get_mirror_template(question)` which inspects
    the question.
    """
    system = SYSTEM_PROMPT_FR if language == "fr" else SYSTEM_PROMPT_EN
    return ChatPromptTemplate.from_messages(
        [
            ("system", system),
            (
                "human",
                """Contexte récupéré:
{context}

Question: {question}

Réponse:""",
            ),
        ]
    )


def get_mirror_template(question: str) -> tuple[str, ChatPromptTemplate]:
    """Pick the right template based on the question's language.

    Returns (language_code, prompt_template). Use this in the chain
    instead of the legacy `get_qa_template(language=...)` so the LLM
    answers in the user's language without a hard-coded choice.
    """
    from src.rag.language import detect_language

    lang = detect_language(question)
    return lang, get_qa_template(language=lang)


__all__ = [
    "SYSTEM_PROMPT_FR",
    "SYSTEM_PROMPT_EN",
    "SYSTEM_PROMPT_MIRROR",
    "get_qa_template",
    "get_mirror_template",
]

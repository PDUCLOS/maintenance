"""QA prompt template for the LangChain RAG chain.

The template is loaded from the .txt files in this directory at import
time, so changes to the prompt text are picked up on next process start.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=1)
def _read(name: str) -> str:
    return (_PROMPTS_DIR / name).read_text(encoding="utf-8")


def get_qa_template(language: str = "fr") -> ChatPromptTemplate:
    """Return the LangChain ChatPromptTemplate for the RAG chain.

    Args:
        language: "fr" or "en" (default "fr").
    """
    system = _read(f"system_{language}.txt")
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


__all__ = ["get_qa_template"]

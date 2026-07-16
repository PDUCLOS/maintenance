"""RAG chain (LangChain LCEL).

Pipeline:
    question -> retrieve -> format context -> prompt -> LLM -> answer

The chain is a simple "format context then prompt then LLM" LCEL chain.
We deliberately do the retrieval OUTSIDE the chain (in `query` and
`stream`) so we can return the retrieved sources alongside the answer
in the RAGResponse — putting retrieval inside the chain would lose
visibility on which chunks were used.

Streaming is supported end-to-end (token by token) via .stream().

Bilingual handling (FR/EN)
---------------------------
The corpus is mostly English (Schaeffler/SKF bearing catalogues) with some French (NTN-SNR).
The user may ask in French or English. We use **mirror response**:
the LLM answers in the same language as the question, while keeping
the source citations in their original language. The system prompt
(`SYSTEM_PROMPT_MIRROR` in `qa_template.py`) is bilingual on purpose so
the LLM sees one consistent persona with the mirror rule reinforced.

Language detection: `src.rag.language.detect_language(question)` —
lightweight heuristic, no extra dep. Used in `query()`/`stream()` to
pick the right per-call prompt template.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from src.config import settings
from src.rag.embeddings import Embedder
from src.rag.language import detect_language
from src.rag.llm import MLXChatModel
from src.rag.prompts.qa_template import get_qa_template
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievedChunk
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger
from src.utils.timing import timed


@dataclass
class RAGResponse:
    """RAG answer with its retrieved sources and detected language."""

    answer: str
    sources: list[RetrievedChunk]
    language: str  # "fr" or "en" — the language of the answer


class RAGChain:
    """The end-to-end RAG chain. Singleton per process (LLM is heavy)."""

    _instance: RAGChain | None = None

    def __init__(self, language: str = "fr") -> None:
        """`language` is the DEFAULT language used only when detection
        fails (empty/unrecognisable question). For real queries the
        actual language is auto-detected per call."""
        self.llm = MLXChatModel()
        self.embedder = Embedder()
        self.vectorstore = VectorStore()
        self.retriever = HybridRetriever(self.vectorstore, self.embedder)
        self.default_language = language
        # We don't build a fixed prompt anymore — the chain is rebuilt
        # per call with the right template. _chain becomes a callable
        # that takes a (prompt, dict) pair.
        self._llm = self.llm
        self._parser = StrOutputParser()

    @classmethod
    def get(cls) -> RAGChain:
        """Process-wide singleton (avoids reloading the LLM on every call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton (used by tests after a vectorstore reset)."""
        cls._instance = None

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Render retrieved documents as a single context string."""
        if not chunks:
            return "(no relevant source found — the knowledge base is empty or the question is out of scope)"
        lines = []
        for i, c in enumerate(chunks, start=1):
            # Use neutral labels in the user-facing context so the LLM
            # doesn't echo internal jargon ("retrieval", "chunk") in its
            # answer. The metadata is preserved via the source label
            # and a relevance score, both of which read naturally.
            lines.append(
                f"[{i}] (source: {c.source}, relevance: {c.score:.2f})\n{c.text}"
            )
        return "\n\n---\n\n".join(lines)

    def _build_invoke(self, language: str):
        """Build a per-call LCEL chain for a specific language.

        Returns a callable that takes ``{"context", "question"}`` and
        returns the LLM's string answer.
        """
        prompt = get_qa_template(language=language)
        return RunnablePassthrough() | prompt | self._llm | self._parser

    @timed
    def query(self, question: str, top_k: int = settings.retriever_top_k) -> RAGResponse:
        """Run the chain. Returns the answer, the retrieved sources, and
        the detected language of the answer."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        language = detect_language(question)
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._format_context(chunks)
        chain = self._build_invoke(language)
        answer = chain.invoke({"context": context, "question": question})
        logger.info(
            "RAG query ok (lang={}, chunks={}, answer_len={})",
            language, len(chunks), len(answer),
        )
        return RAGResponse(answer=answer, sources=chunks, language=language)

    def stream(
        self,
        question: str,
        top_k: int = settings.retriever_top_k,
    ) -> Iterator[str]:
        """Stream the answer token by token. The retrieved sources are
        not returned in the stream (call `query` for the full response)."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        language = detect_language(question)
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._format_context(chunks)
        chain = self._build_invoke(language)
        yield from chain.stream({"context": context, "question": question})


__all__ = ["RAGChain", "RAGResponse"]

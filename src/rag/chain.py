"""RAG chain (LangChain LCEL).

Pipeline:
    question -> retrieve -> format context -> prompt -> LLM -> answer

The chain is a simple "format context then prompt then LLM" LCEL chain.
We deliberately do the retrieval OUTSIDE the chain (in `query` and
`stream`) so we can return the retrieved sources alongside the answer
in the RAGResponse — putting retrieval inside the chain would lose
visibility on which chunks were used.

Streaming is supported end-to-end (token by token) via .stream().
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnableLambda, RunnablePassthrough

from src.config import settings  # FIX Bug #1: this import was missing
from src.rag.embeddings import Embedder
from src.rag.llm import MLXChatModel
from src.rag.prompts.qa_template import get_qa_template
from src.rag.retriever import HybridRetriever
from src.rag.types import RetrievedChunk
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger
from src.utils.timing import timed


@dataclass
class RAGResponse:
    """RAG answer with its retrieved sources."""

    answer: str
    sources: list[RetrievedChunk]


class RAGChain:
    """The end-to-end RAG chain. Singleton per process (LLM is heavy)."""

    _instance: "RAGChain | None" = None

    def __init__(self, language: str = "fr") -> None:
        self.llm = MLXChatModel()
        self.embedder = Embedder()
        self.vectorstore = VectorStore()
        self.retriever = HybridRetriever(self.vectorstore, self.embedder)
        self.prompt = get_qa_template(language=language)
        self._chain: Runnable = self._build_chain()

    @classmethod
    def get(cls) -> "RAGChain":
        """Process-wide singleton (avoids reloading Mistral 7B on every call)."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @classmethod
    def reset_singleton(cls) -> None:
        """Reset the singleton (used by tests after a vectorstore reset)."""
        cls._instance = None

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks as a single context string."""
        if not chunks:
            return "(aucun contexte récupéré — la base de connaissances est vide ou la question est hors-scope)"
        lines = []
        for i, c in enumerate(chunks, start=1):
            lines.append(
                f"[{i}] (source={c.source}, retrieval={c.retrieval_method}, score={c.score:.4f})\n{c.text}"
            )
        return "\n\n---\n\n".join(lines)

    def _build_chain(self) -> Runnable:
        """Build a simple LCEL chain.

        The chain takes a dict ``{"context": str, "question": str}`` (the
        prompt variables) and pipes through prompt -> LLM -> string output.
        Retrieval is done by the caller (in `query`/`stream`), NOT inside
        the chain, so we can return the retrieved sources alongside the
        answer in the RAGResponse.

        The RunnablePassthrough is identity here — it just forwards the
        input dict unchanged. We keep it explicit for readability.
        """
        chain: Runnable = (
            RunnablePassthrough()
            | self.prompt
            | self.llm
            | StrOutputParser()
        )
        return chain

    @timed
    def query(self, question: str, top_k: int = settings.retriever_top_k) -> RAGResponse:
        """Run the chain. Returns the answer plus its retrieved sources.

        We do the retrieval explicitly (once) so we can return the sources
        in the RAGResponse and pass them to the chain's context.
        """
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._format_context(chunks)
        # FIX Bug #2: pass a dict that matches the prompt's variables
        # (context + question), not a string. The chain doesn't do
        # retrieval itself — retrieval is owned by `query`/`stream`.
        answer = self._chain.invoke({"context": context, "question": question})
        logger.info(
            "RAG query ok (chunks={}, answer_len={})", len(chunks), len(answer)
        )
        return RAGResponse(answer=answer, sources=chunks)

    def stream(
        self,
        question: str,
        top_k: int = settings.retriever_top_k,
    ) -> Iterator[str]:
        """Stream the answer token by token (retrieval upfront, as in `query`)."""
        if not question or not question.strip():
            raise ValueError("question must not be empty")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._format_context(chunks)
        for token in self._chain.stream({"context": context, "question": question}):
            yield token


__all__ = ["RAGChain", "RAGResponse"]

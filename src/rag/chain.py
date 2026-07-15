"""RAG chain (LangChain LCEL).

Pipeline:
    question -> retriever -> context formatter -> prompt -> LLM -> answer

Streaming is supported end-to-end (chunk by chunk).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable, RunnablePassthrough

from src.rag.embeddings import Embedder
from src.rag.llm import MLXChatModel
from src.rag.prompts.qa_template import get_qa_template
from src.rag.retriever import HybridRetriever, RetrievedChunk
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

    def _format_context(self, chunks: list[RetrievedChunk]) -> str:
        """Render retrieved chunks as a single context string."""
        if not chunks:
            return "(aucun contexte récupéré)"
        lines = []
        for i, c in enumerate(chunks, start=1):
            lines.append(f"[{i}] (source={c.source}, score={c.score:.3f})\n{c.text}")
        return "\n\n---\n\n".join(lines)

    def _build_chain(self) -> Runnable:
        """Build the LCEL chain: retriever -> context -> prompt -> llm -> str."""
        raise NotImplementedError(
            "LCEL chain: to be implemented in W2. Pattern: "
            "{'context': retriever_fn, 'question': RunnablePassthrough()} "
            "| prompt | self.llm | StrOutputParser()."
        )

    @timed
    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Run the chain. Returns the answer plus its retrieved sources."""
        if not question.strip():
            raise ValueError("question must not be empty")
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._format_context(chunks)
        answer = self._chain.invoke({"question": question, "context": context})
        logger.info("RAG query ok (chunks={}, answer_len={})", len(chunks), len(answer))
        return RAGResponse(answer=answer, sources=chunks)

    def stream(self, question: str, top_k: int = 5) -> Iterator[str]:
        """Stream the answer token by token (still uses full retrieval upfront)."""
        chunks = self.retriever.retrieve(question, top_k=top_k)
        context = self._format_context(chunks)
        for token in self._chain.stream({"question": question, "context": context}):
            yield token


__all__ = ["RAGChain", "RAGResponse"]

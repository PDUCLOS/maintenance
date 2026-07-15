"""Hybrid retriever: BM25 (lexical) + dense (embedding) fused with RRF.

Why hybrid? The CMAPSS readme and technical PDFs are dense prose, but
sensor names, unit IDs and column names are best matched lexically. Pure
embedding retrieval misses "FD001" and exact sensor identifiers. BM25
covers that, and Reciprocal Rank Fusion combines both rankings without
requiring score calibration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import settings
from src.rag.embeddings import Embedder
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its fused score and original source."""

    chunk_id: str
    text: str
    source: str
    metadata: dict[str, str]
    score: float
    retrieval_method: str  # "dense", "bm25", or "rrf"


@dataclass
class HybridRetriever:
    """Combines ChromaDB (dense) with an in-memory BM25 index.

    The BM25 index is rebuilt on first use from a snapshot of the Chroma
    collection. If the collection is rebuilt, call `invalidate_bm25_cache`
    before the next query.
    """

    vectorstore: VectorStore
    embedder: Embedder
    _bm25: BM25Retriever | None = field(default=None, init=False, repr=False)
    _bm25_dirty: bool = field(default=True, init=False, repr=False)

    def invalidate_bm25_cache(self) -> None:
        self._bm25_dirty = True
        self._bm25 = None

    def _ensure_bm25(self) -> BM25Retriever:
        if self._bm25 is None or self._bm25_dirty:
            docs = self._fetch_all_documents()
            if not docs:
                raise RuntimeError(
                    "Chroma collection is empty. Run 'make ingest' to populate it "
                    "before querying."
                )
            self._bm25 = BM25Retriever.from_documents(docs)
            self._bm25.k = settings.retriever_top_k
            self._bm25_dirty = False
            logger.info("BM25 index rebuilt over {} documents", len(docs))
        return self._bm25

    def _fetch_all_documents(self) -> list[Document]:
        """Pull every chunk out of Chroma to feed BM25.

        Cheap for our scale (a few thousand chunks max). If the index grows
        past 100k, switch to a persistent BM25 index on disk.
        """
        raise NotImplementedError(
            "Hybrid retriever: to be implemented in W2. Use "
            "self.vectorstore._collection.get(include=['documents','metadatas']) "
            "and convert each row to a langchain Document."
        )

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retriever_top_k,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks via RRF(dense, bm25)."""
        if not settings.hybrid_search:
            return self._dense_only(query, top_k)
        return self._hybrid(query, top_k)

    def _dense_only(self, query: str, top_k: int) -> list[RetrievedChunk]:
        qv = self.embedder.embed([query])[0]
        hits = self.vectorstore.query(qv, top_k=top_k)
        return [
            RetrievedChunk(
                chunk_id=h["id"],
                text=h["text"],
                source=h["metadata"].get("source", "unknown"),
                metadata={k: str(v) for k, v in h["metadata"].items()},
                score=h["score"],
                retrieval_method="dense",
            )
            for h in hits
        ]

    def _hybrid(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Reciprocal Rank Fusion of dense and BM25 rankings."""
        raise NotImplementedError(
            "Hybrid RRF: to be implemented in W2. Get top_k from both retrievers, "
            "compute RRF score = sum(1 / (k + rank)) over both lists, "
            "sort desc, return top_k."
        )


__all__ = ["HybridRetriever", "RetrievedChunk"]

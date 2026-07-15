"""Hybrid retriever: BM25 (lexical) + dense (embedding) fused with RRF.

Why hybrid? The CMAPSS readme and technical PDFs are dense prose, but
sensor names, unit IDs and column names are best matched lexically. Pure
embedding retrieval misses "FD001" and exact sensor identifiers. BM25
covers that, and Reciprocal Rank Fusion combines both rankings without
requiring score calibration.

Reference: Cormack et al. (2009), "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR.
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

# Standard RRF constant (k0 in the paper). 60 is the commonly-cited value
# that performs best across most corpora.
_RRF_K0 = 60


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
        result = self.vectorstore._collection.get(include=["documents", "metadatas"])
        ids = result.get("ids") or []
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        docs: list[Document] = []
        for i, doc_id in enumerate(ids):
            text = documents[i] if i < len(documents) and documents[i] else ""
            meta = metadatas[i] if i < len(metadatas) and metadatas[i] else {}
            # BM25Retriever expects a str page_content and a dict metadata.
            # Strip non-string values from metadata to avoid surprises.
            clean_meta = {str(k): str(v) for k, v in meta.items()}
            clean_meta.setdefault("chunk_id", str(doc_id))
            docs.append(Document(page_content=text, metadata=clean_meta))
        return docs

    def retrieve(
        self,
        query: str,
        top_k: int = settings.retriever_top_k,
    ) -> list[RetrievedChunk]:
        """Retrieve top_k chunks via RRF(dense, bm25) or dense-only."""
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
        """Reciprocal Rank Fusion of dense and BM25 rankings.

        Each chunk's final score is the sum of 1/(k0+rank) over both lists.
        Chunks that appear in only one list still get partial credit.
        The retrieval_method field records where each chunk was found.
        """
        # 1. Dense retrieval
        qv = self.embedder.embed([query])[0]
        dense_hits = self.vectorstore.query(qv, top_k=top_k)
        # Build a map chunk_id -> {chunk, rrf}
        chunk_map: dict[str, dict[str, Any]] = {}

        for rank, hit in enumerate(dense_hits):
            cid = hit["id"]
            chunk_map[cid] = {
                "chunk": RetrievedChunk(
                    chunk_id=hit["id"],
                    text=hit["text"],
                    source=hit["metadata"].get("source", "unknown"),
                    metadata={k: str(v) for k, v in hit["metadata"].items()},
                    score=0.0,
                    retrieval_method="dense",
                ),
                "rrf": 1.0 / (_RRF_K0 + rank + 1),
            }

        # 2. BM25 retrieval
        bm25 = self._ensure_bm25()
        bm25.k = top_k
        # langchain_community BM25Retriever exposes get_relevant_documents
        # (older API) and invokes LangChain Runnable interface in newer
        # versions. Try both.
        try:
            bm25_docs = bm25.invoke(query)  # type: ignore[attr-defined]
        except AttributeError:
            bm25_docs = bm25.get_relevant_documents(query)

        for rank, doc in enumerate(bm25_docs):
            cid = doc.metadata.get("chunk_id", "")
            if not cid:
                # Skip docs without a chunk_id (shouldn't happen, but be safe).
                logger.warning("BM25 doc missing chunk_id, skipping: {}", doc.metadata)
                continue
            rrf_contrib = 1.0 / (_RRF_K0 + rank + 1)
            if cid in chunk_map:
                chunk_map[cid]["rrf"] += rrf_contrib
                if chunk_map[cid]["chunk"].retrieval_method == "dense":
                    chunk_map[cid]["chunk"].retrieval_method = "rrf"
            else:
                chunk_map[cid] = {
                    "chunk": RetrievedChunk(
                        chunk_id=cid,
                        text=doc.page_content,
                        source=doc.metadata.get("source", "unknown"),
                        metadata={k: str(v) for k, v in doc.metadata.items()},
                        score=0.0,
                        retrieval_method="bm25",
                    ),
                    "rrf": rrf_contrib,
                }

        # 3. Sort by RRF score, take top_k
        sorted_items = sorted(chunk_map.values(), key=lambda x: x["rrf"], reverse=True)[:top_k]
        for item in sorted_items:
            item["chunk"].score = item["rrf"]
        return [item["chunk"] for item in sorted_items]


__all__ = ["HybridRetriever", "RetrievedChunk"]

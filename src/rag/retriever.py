"""Hybrid retriever: BM25 (lexical) + dense (embedding) fused with RRF.

Why hybrid? Technical PDFs (bearing catalogues) are dense prose, but
sensor names, unit IDs and column names are best matched lexically. Pure
embedding retrieval misses "FD001" and exact sensor identifiers. BM25
covers that, and Reciprocal Rank Fusion combines both rankings without
requiring score calibration.

Reference: Cormack et al. (2009), "Reciprocal Rank Fusion outperforms
Condorcet and individual Rank Learning Methods", SIGIR.

Optionally, the retriever is paired with a cross-encoder reranker
(see `src.rag.reranker`). When `settings.reranker_enabled` is true,
the retriever over-fetches candidates and the reranker trims to top_k.

BM25 caching:
    The BM25 index is expensive to build (~30s for 11k chunks on M5
    Pro). We pickle the whole BM25Retriever to `data/processed/bm25_cache.pkl`
    on first build, keyed by a SHA256 of the corpus (chunks ids + first
    200 chars of each text). On subsequent startups, if the corpus hash
    matches, we load from disk in < 1s instead of rebuilding.
    `invalidate_bm25_cache()` drops both the in-memory and on-disk copy.
"""

from __future__ import annotations

import hashlib
import pickle
from dataclasses import dataclass, field
from typing import Any

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document

from src.config import settings
from src.rag.embeddings import Embedder
from src.rag.reranker import (
    RERANK_OVERFETCH,
    Reranker,
)
from src.rag.types import RetrievedChunk  # re-export for back-compat
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger

# Standard RRF constant (k0 in the paper). 60 is the commonly-cited value
# that performs best across most corpora.
_RRF_K0 = 60

# Persistent BM25 cache location. Sits next to the chunks.jsonl file so
# it's automatically cleaned up if the user wipes data/processed.
_BM25_CACHE_PATH = settings.processed_data_dir / "bm25_cache.pkl"


@dataclass
class HybridRetriever:
    """Combines ChromaDB (dense) with an in-memory BM25 index.

    The BM25 index is rebuilt on first use from a snapshot of the Chroma
    collection. If the collection is rebuilt, call `invalidate_bm25_cache`
    before the next query.
    """

    vectorstore: VectorStore
    embedder: Embedder
    reranker: Reranker | None = None
    _bm25: BM25Retriever | None = field(default=None, init=False, repr=False)
    _bm25_dirty: bool = field(default=True, init=False, repr=False)

    def __post_init__(self) -> None:
        if settings.reranker_enabled:
            self.reranker = Reranker()

    def invalidate_bm25_cache(self) -> None:
        self._bm25_dirty = True
        self._bm25 = None
        # Also drop the on-disk cache so the next call rebuilds from
        # the current chroma collection state.
        if _BM25_CACHE_PATH.is_file():
            _BM25_CACHE_PATH.unlink()
            logger.info("BM25 cache file removed: {}", _BM25_CACHE_PATH)

    @staticmethod
    def _corpus_hash(docs: list[Document]) -> str:
        """Hash the corpus so a stale cache is detectable.

        We hash (chunk_id + first 200 chars of text) for every doc. This
        is robust to re-ingestion with the same content (cache hit) and
        to any change in text or chunking (cache miss → rebuild).
        200 chars is enough to catch every meaningful content change
        while keeping the hash fast.
        """
        h = hashlib.sha256()
        for d in docs:
            h.update(d.metadata.get("chunk_id", "").encode("utf-8"))
            h.update(b"\x00")
            h.update(d.page_content[:200].encode("utf-8", errors="ignore"))
            h.update(b"\x1f")
        return h.hexdigest()[:16]  # 16 hex chars = 64 bits, plenty for this scale

    def _load_bm25_from_cache(self, corpus_hash: str) -> BM25Retriever | None:
        """Try to load a cached BM25Retriever from disk.

        Returns None if the file doesn't exist, is malformed, or the
        stored hash doesn't match the current corpus. Errors are
        logged at debug level — cache failure is never fatal, we just
        rebuild.
        """
        if not _BM25_CACHE_PATH.is_file():
            return None
        try:
            blob = pickle.loads(_BM25_CACHE_PATH.read_bytes())
            if not isinstance(blob, dict):
                logger.warning("BM25 cache has unexpected shape, ignoring")
                return None
            if blob.get("corpus_hash") != corpus_hash:
                logger.info(
                    "BM25 cache hash mismatch (cached={}, current={}), rebuild needed",
                    blob.get("corpus_hash"),
                    corpus_hash,
                )
                return None
            bm25: BM25Retriever = blob["bm25"]
            bm25.k = settings.retriever_top_k
            logger.info(
                "BM25 cache hit: loaded {} docs from {}",
                blob.get("n_docs", "?"),
                _BM25_CACHE_PATH,
            )
            return bm25
        except Exception as e:
            # Corrupt pickle, missing keys, version mismatch, anything.
            # Cache is best-effort — fall through to rebuild.
            logger.warning("BM25 cache load failed ({}), will rebuild", e)
            return None

    def _save_bm25_to_cache(
        self, bm25: BM25Retriever, docs: list[Document], corpus_hash: str
    ) -> None:
        """Persist the BM25Retriever + corpus hash to disk.

        We wrap the retriever + a small metadata sidecar in a dict so
        we can detect hash mismatches on load (vs. blindly trusting
        whatever pickle gave us back).
        """
        _BM25_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        blob = {
            "corpus_hash": corpus_hash,
            "n_docs": len(docs),
            "bm25": bm25,
        }
        _BM25_CACHE_PATH.write_bytes(pickle.dumps(blob, protocol=pickle.HIGHEST_PROTOCOL))
        size_mb = _BM25_CACHE_PATH.stat().st_size / 1048576
        logger.info(
            "BM25 cache written: {} docs, {:.1f} MB → {}",
            len(docs),
            size_mb,
            _BM25_CACHE_PATH,
        )

    def _ensure_bm25(self) -> BM25Retriever:
        if self._bm25 is not None and not self._bm25_dirty:
            return self._bm25

        docs = self._fetch_all_documents()
        if not docs:
            raise RuntimeError(
                "Chroma collection is empty. Run 'make ingest' to populate it " "before querying."
            )
        corpus_hash = self._corpus_hash(docs)

        # Try cache first (fast path, < 1s)
        cached = self._load_bm25_from_cache(corpus_hash)
        if cached is not None:
            self._bm25 = cached
            self._bm25_dirty = False
            return self._bm25

        # Slow path: rebuild from scratch (~30s for 11k chunks)
        logger.info("BM25 index rebuilding over {} documents...", len(docs))
        self._bm25 = BM25Retriever.from_documents(docs)
        self._bm25.k = settings.retriever_top_k
        self._bm25_dirty = False
        # Persist for next start. If this fails, the retriever still
        # works in-process — we just lose the speedup next time.
        try:
            self._save_bm25_to_cache(self._bm25, docs, corpus_hash)
        except Exception as e:
            logger.warning("BM25 cache write failed ({}), in-memory index only", e)
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
        """Retrieve top_k chunks via RRF(dense, bm25) or dense-only,
        optionally reranked by a cross-encoder.

        FIX Bug #3: when the reranker is enabled, we over-fetch
        (top_k * RERANK_OVERFETCH) candidates, then the reranker trims
        to top_k. Previously the over-fetch was missing and the reranker
        was never triggered.
        """
        # Decide how many candidates to fetch from the base retrievers
        fetch_k = max(top_k, top_k * RERANK_OVERFETCH) if self.reranker is not None else top_k

        if not settings.hybrid_search:
            chunks = self._dense_only(query, fetch_k)
        else:
            chunks = self._hybrid(query, fetch_k)

        # If the reranker is enabled, rerank and trim to top_k
        if self.reranker is not None and len(chunks) > top_k:
            chunks = self.reranker.rerank(query, chunks, top_n=top_k)
        return chunks

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

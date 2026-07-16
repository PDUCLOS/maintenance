"""Cross-encoder reranker.

After the initial retrieval (BM25 + dense fused via RRF), we over-fetch
candidates and rerank them with a cross-encoder. The cross-encoder takes
(query, document) pairs and produces a calibrated relevance score —
much more accurate than either dense cosine or BM25 alone, at the cost
of an extra forward pass per pair.

Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- 22M params, fast on M-series
- Trained on MS MARCO passage ranking
- Output: float score (higher = more relevant)

The reranker is OPT-IN via `settings.reranker_enabled`. When enabled,
the retriever fetches `top_k * 3` candidates and the reranker trims to
`top_k`. This gives the reranker room to reorder.
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING, Any

from src.config import settings
from src.utils.logger import logger

if TYPE_CHECKING:
    # Imported only for type hints — avoids the runtime circular import
    # between retriever.py and reranker.py.
    from src.rag.types import RetrievedChunk

# Over-fetch multiplier: we ask the retriever for this x top_k candidates,
# then the reranker trims to top_k. Higher = more reranker work but better recall.
RERANK_OVERFETCH = 3


class Reranker:
    """Cross-encoder reranker with lazy model loading (MPS-accelerated)."""

    _model: Any = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self, model_name: str | None = None) -> None:
        self.model_name = model_name or settings.reranker_model

    def _load(self) -> None:
        if Reranker._model is not None:
            return
        with Reranker._lock:
            if Reranker._model is not None:
                return
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: {}", self.model_name)
            # device='mps' is the Apple Silicon GPU path.
            Reranker._model = CrossEncoder(self.model_name, device="mps")
            logger.info("Reranker ready")

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_n: int,
    ) -> list[RetrievedChunk]:
        """Rerank `chunks` for `query`, return top_n by cross-encoder score.

        If `chunks` is empty or has ≤ top_n items, returns as-is (or trimmed).
        The returned chunks have their `score` field replaced by the
        cross-encoder score, and `retrieval_method` updated to "reranked".
        """
        # Local import to avoid the circular dependency at module level
        from src.rag.types import RetrievedChunk

        if not chunks:
            return []
        if len(chunks) <= top_n:
            return chunks

        self._load()
        assert Reranker._model is not None

        # CrossEncoder expects an iterable of (query, doc) pairs
        pairs = [(query, c.text) for c in chunks]
        scores = Reranker._model.predict(pairs, show_progress_bar=False)

        # Sort by score descending, take top_n
        scored = list(zip(chunks, scores, strict=False))
        scored.sort(key=lambda x: float(x[1]), reverse=True)
        top = scored[:top_n]

        # Rebuild RetrievedChunks with the new score
        out: list[RetrievedChunk] = []
        for c, s in top:
            out.append(
                RetrievedChunk(
                    chunk_id=c.chunk_id,
                    text=c.text,
                    source=c.source,
                    metadata=c.metadata,
                    score=float(s),
                    retrieval_method="reranked",
                )
            )
        logger.debug("Reranked {} -> top {}", len(chunks), top_n)
        return out


__all__ = ["RERANK_OVERFETCH", "Reranker"]

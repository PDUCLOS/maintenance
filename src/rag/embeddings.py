"""Embedding model wrapper.

We use sentence-transformers with `device="mps"` (Apple Metal). This is the
most stable path for embeddings on Apple Silicon as of 2026 — it benefits
from GPU acceleration without the fragility of rolling our own MLX
embedding inference. The LLM side uses MLX (see `llm.py`).

Why not pure MLX for embeddings?
- The MLX ecosystem for embedding models is less mature than for LLMs.
- sentence-transformers + MPS gives the same throughput, with battle-tested
  model cards (bge, nomic, e5) and built-in batching.
- Migrating to MLX later is a 50-line swap if needed.

If mlx is required for embeddings too, swap the body of `Embedder.embed`
to use `mlx.core` + a HF model converted to MLX format.
"""

from __future__ import annotations

import threading
from typing import Any

from src.config import settings
from src.utils.logger import logger


class Embedder:
    """Thin wrapper around sentence-transformers with MPS acceleration.

    Lazy-loads the model on first call to keep import time low. Thread-safe
    (sentence-transformers encoding is thread-safe per its docs).
    """

    _model: Any = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._repo = settings.mlx_embed_repo
        self._dim = settings.embed_dim

    def _load(self) -> None:
        if Embedder._model is not None:
            return
        with Embedder._lock:
            if Embedder._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model {}", self._repo)
            Embedder._model = SentenceTransformer(self._repo, device="mps")
            logger.info("Embedding model ready (dim={})", self._model.get_sentence_embedding_dimension())

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of float vectors."""
        self._load()
        assert Embedder._model is not None
        if not texts:
            return []
        vectors = Embedder._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine sim == dot product
            show_progress_bar=False,
        )
        return vectors.tolist()

    @property
    def dim(self) -> int:
        """Embedding vector dimension (set after first load, or from settings)."""
        if Embedder._model is not None:
            return int(Embedder._model.get_sentence_embedding_dimension())
        return self._dim


__all__ = ["Embedder"]

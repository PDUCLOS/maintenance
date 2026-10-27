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

Portable path: with `EMBED_PROVIDER=ollama` the same bge-m3 model is
served by Ollama (`/api/embed`, GPU on Windows/Linux) — no torch needed.
Vectors are L2-normalised either way, so the Chroma index is the same.
"""

from __future__ import annotations

import threading
from typing import Any

from langchain_core.embeddings import Embeddings

from src.config import settings
from src.utils.logger import logger


class Embedder:
    """Thin wrapper around sentence-transformers with MPS acceleration.

    Lazy-loads the model on first call to keep import time low. Thread-safe
    (sentence-transformers encoding is thread-safe per its docs).
    """

    _model: Any = None
    _lock: threading.Lock = threading.Lock()

    # Ollama batch size: one HTTP call per batch keeps memory flat on
    # large ingests while still amortising the request overhead.
    _OLLAMA_BATCH = 64

    def __init__(self) -> None:
        self._repo = settings.mlx_embed_repo
        self._dim = settings.embed_dim
        self._provider = settings.embed_provider

    def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        import httpx
        import numpy as np

        out: list[list[float]] = []
        with httpx.Client(timeout=settings.ollama_timeout_seconds) as client:
            for start in range(0, len(texts), self._OLLAMA_BATCH):
                batch = texts[start : start + self._OLLAMA_BATCH]
                resp = client.post(
                    f"{settings.ollama_base_url}/api/embed",
                    json={"model": settings.ollama_embed_model, "input": batch},
                )
                resp.raise_for_status()
                vecs = np.asarray(resp.json()["embeddings"], dtype=np.float32)
                vecs /= np.linalg.norm(vecs, axis=1, keepdims=True).clip(min=1e-12)
                out.extend(vecs.tolist())
                if len(texts) > 50 and (start // self._OLLAMA_BATCH) % 20 == 0:
                    logger.info("Embedded {}/{} chunks (ollama)", start + len(batch), len(texts))
        return out

    def _load(self) -> None:
        if Embedder._model is not None:
            return
        with Embedder._lock:
            if Embedder._model is not None:
                return
            from sentence_transformers import SentenceTransformer

            logger.info("Loading embedding model {}", self._repo)
            Embedder._model = SentenceTransformer(
                self._repo, device=settings.resolve_torch_device()
            )
            logger.info(
                "Embedding model ready (dim={})", self._model.get_sentence_embedding_dimension()
            )

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts. Returns a list of float vectors."""
        if self._provider == "ollama":
            return self._embed_ollama(texts) if texts else []
        self._load()
        assert Embedder._model is not None
        if not texts:
            return []
        # Show a tqdm progress bar for non-trivial batches (the bge-m3
        # ingest takes ~20 min on M5 Pro without feedback — UX disaster).
        # Below 50 texts the bar is noise.
        show_bar = len(texts) > 50
        if show_bar:
            logger.info("Embedding {} chunks (~20 min on M5 Pro)...", len(texts))
        vectors = Embedder._model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,  # cosine sim == dot product
            show_progress_bar=show_bar,
        )
        return vectors.tolist()

    @property
    def dim(self) -> int:
        """Embedding vector dimension (set after first load, or from settings)."""
        if Embedder._model is not None:
            return int(Embedder._model.get_sentence_embedding_dimension())
        return self._dim


class LangChainEmbedder(Embeddings):
    """Adapts `Embedder` to the `langchain_core.embeddings.Embeddings` ABC.

    Needed to pass our local embedding model into RAGAS's `evaluate(...,
    embeddings=...)` — without this, RAGAS defaults to OpenAI embeddings,
    which contradicts the "100% local, no API key" pitch.
    """

    def __init__(self, embedder: Embedder | None = None) -> None:
        self._embedder = embedder or Embedder()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embedder.embed(texts)

    def embed_query(self, text: str) -> list[float]:
        return self._embedder.embed([text])[0]


__all__ = ["Embedder", "LangChainEmbedder"]

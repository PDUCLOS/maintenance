"""ChromaDB vector store wrapper.

We connect to a Chroma server running in Docker (see docker-compose.yml)
via HTTP. The client/server split lets us rebuild the index from a script
without spinning up the Python process.
"""

from __future__ import annotations

from typing import Any

import chromadb

from src.config import settings
from src.ingestion.chunker import Chunk
from src.utils.logger import logger


class VectorStore:
    """Thin wrapper around chromadb.HttpClient.

    Uses cosine similarity (HNSW default) which matches our normalized
    embeddings. The collection is created on first access if it doesn't
    exist.
    """

    def __init__(self) -> None:
        self._client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
        )
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "Chroma collection {} ready (count={})",
            settings.chroma_collection,
            self._collection.count(),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Add or update chunks. Chunks and vectors must be aligned."""
        if not chunks:
            logger.warning("upsert called with empty chunks list")
            return
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunks/vectors length mismatch: {len(chunks)} vs {len(vectors)}"
            )
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=vectors,
            documents=[c.text for c in chunks],
            metadatas=[{**c.metadata, "source": c.source} for c in chunks],
        )
        logger.info("Upserted {} chunks", len(chunks))

    def query(
        self,
        query_vector: list[float],
        top_k: int = settings.retriever_top_k,
        where: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Return top-k chunks by similarity to the query vector."""
        result = self._collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=where,
        )
        # Chroma returns lists-of-lists; flatten to a list of dicts.
        if not result["ids"]:
            return []
        return [
            {
                "id": result["ids"][0][i],
                "text": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
                "score": 1.0 - result["distances"][0][i],  # cosine sim (Chroma returns distance)
            }
            for i in range(len(result["ids"][0]))
        ]

    def count(self) -> int:
        return self._collection.count()

    def reset(self) -> None:
        """Delete and recreate the collection. Irreversible — used by tests."""
        self._client.delete_collection(settings.chroma_collection)
        self._collection = self._client.get_or_create_collection(
            name=settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )


__all__ = ["VectorStore"]

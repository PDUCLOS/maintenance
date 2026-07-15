"""Shared types for the RAG pipeline.

This module exists to break the circular import between
`src.rag.retriever` and `src.rag.reranker` (retriever needs the
reranker class; reranker needs the RetrievedChunk type).

Putting `RetrievedChunk` in a leaf module that both can import
keeps the dependency graph acyclic.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    """A retrieved chunk with its (possibly fused or reranked) score and source."""

    chunk_id: str
    text: str
    source: str
    metadata: dict[str, str]
    score: float
    retrieval_method: str  # "dense", "bm25", "rrf", or "reranked"


__all__ = ["RetrievedChunk"]

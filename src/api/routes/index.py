"""/index/stats endpoint — ChromaDB collection stats for the UI's Index tab.

Exists so the Streamlit UI never has to open its own chromadb.HttpClient:
a second client created from Streamlit's own process/thread crashes with a
segfault (chromadb's posthog telemetry call fails with a TypeError on this
version, and something about triggering that from a non-main thread inside
Streamlit's script-rerun model corrupts the interpreter — same family of
issue as MLX's thread-bound Metal stream elsewhere in this codebase). The
API process already owns one healthy chromadb client; the UI just asks it.
"""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import IndexStatsResponse, SourceChunk
from src.rag.vectorstore import VectorStore

router = APIRouter(prefix="/index", tags=["index"])


@router.get("/stats", response_model=IndexStatsResponse)
def stats(sample_limit: int = 5, source_scan_limit: int = 10000) -> IndexStatsResponse:
    """Collection count, per-source chunk distribution, and a few sample chunks."""
    vs = VectorStore()
    result = vs._collection.get(include=["documents", "metadatas"], limit=source_scan_limit)
    ids = result.get("ids") or []
    documents = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    source_counts: dict[str, int] = {}
    for meta in metadatas:
        src = (meta or {}).get("source", "unknown")
        short = src.split(":")[0] if ":" in src else src
        source_counts[short] = source_counts.get(short, 0) + 1

    sample_chunks = [
        SourceChunk(
            id=ids[i],
            text=documents[i] if i < len(documents) else "",
            source=(metadatas[i] or {}).get("source", "unknown")
            if i < len(metadatas)
            else "unknown",
            score=0.0,
            metadata=metadatas[i] or {} if i < len(metadatas) else {},
        )
        for i in range(min(sample_limit, len(ids)))
    ]

    return IndexStatsResponse(
        collection_count=vs.count(),
        source_counts=source_counts,
        sample_chunks=sample_chunks,
    )

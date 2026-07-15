"""/ingest endpoint — trigger the ingestion pipeline."""

from __future__ import annotations

import time

from fastapi import APIRouter

from src.api.schemas import IngestRequest, IngestResponse
from src.ingestion import pipeline as ingest_pipeline
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
def ingest(req: IngestRequest) -> IngestResponse:
    """Run the full ingestion pipeline (loaders -> chunker -> embed -> Chroma)."""
    t0 = time.perf_counter()
    if req.force_rebuild:
        VectorStore().reset()
    ingest_pipeline.run()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    n = VectorStore().count()
    logger.info("/ingest ok (chunks={}, latency_ms={:.1f})", n, elapsed_ms)
    return IngestResponse(
        chunks_indexed=n,
        duration_ms=elapsed_ms,
        collection=__import__("src.config", fromlist=["settings"]).settings.chroma_collection,
    )

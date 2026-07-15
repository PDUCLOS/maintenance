"""/query endpoint — run the RAG chain (or the agent)."""

from __future__ import annotations

from fastapi import APIRouter

from src.api.schemas import QueryRequest, QueryResponse, SourceChunk
from src.rag.chain import RAGChain
from src.rag.retriever import RetrievedChunk
from src.utils.logger import logger

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    """Run a question through the RAG chain.

    Note: streaming is not yet supported over HTTP. Use the local Python
    API (`RAGChain.stream()`) or the Streamlit UI for token-by-token UX.
    """
    import time

    t0 = time.perf_counter()
    chain = RAGChain.get()
    rag_response = chain.query(req.question, top_k=req.top_k)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info("/query ok (latency_ms={:.1f}, chunks={})", elapsed_ms, len(rag_response.sources))
    return QueryResponse(
        answer=rag_response.answer,
        sources=[_to_source(c) for c in rag_response.sources],
        latency_ms=elapsed_ms,
    )


def _to_source(c: RetrievedChunk) -> SourceChunk:
    return SourceChunk(
        id=c.chunk_id,
        text=c.text,
        source=c.source,
        score=c.score,
        metadata=c.metadata,
    )

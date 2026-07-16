"""/query endpoint — run the RAG chain (or the agent)."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from src.api.schemas import QueryRequest, QueryResponse, SourceChunk
from src.config import settings
from src.rag.chain import RAGChain
from src.rag.types import RetrievedChunk
from src.utils.logger import logger

router = APIRouter(prefix="/query", tags=["query"])

# A dedicated executor for /query so we can put a hard timeout on the
# whole chain.query() call. The chain internally uses its own executor
# (MLX_EXECUTOR) for the LLM — the two layers of executors are fine.
_QUERY_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="query")


@router.post("", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse | JSONResponse:
    """Run a question through the RAG chain.

    Hard timeout: if `chain.query()` doesn't return within
    `settings.query_timeout_seconds`, the request returns 504. The
    underlying LLM thread is NOT killed (Python can't safely do that
    from outside the thread), but the API stays responsive.

    Note: streaming is not yet supported over HTTP. Use the local
    Python API (`RAGChain.stream()`) or the Streamlit UI for
    token-by-token UX.
    """
    t0 = time.perf_counter()
    chain = RAGChain.get()
    try:
        future = _QUERY_EXECUTOR.submit(chain.query, req.question, top_k=req.top_k)
        rag_response = future.result(timeout=settings.query_timeout_seconds)
    except FutureTimeout:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.error(
            "/query TIMEOUT after {:.1f}ms (limit={}s) — LLM is still running in background",
            elapsed_ms,
            settings.query_timeout_seconds,
        )
        return JSONResponse(
            status_code=504,
            content={
                "error": "query_timeout",
                "message": (
                    f"The RAG chain did not return within "
                    f"{settings.query_timeout_seconds}s. The LLM is still "
                    f"running in the background — try again in a few seconds."
                ),
                "elapsed_ms": elapsed_ms,
            },
        )
    except Exception as e:
        elapsed_ms = (time.perf_counter() - t0) * 1000
        logger.exception("/query FAILED after {:.1f}ms: {}", elapsed_ms, e)
        return JSONResponse(
            status_code=500,
            content={
                "error": "internal_error",
                "message": str(e),
                "elapsed_ms": elapsed_ms,
            },
        )

    elapsed_ms = (time.perf_counter() - t0) * 1000
    logger.info(
        "/query ok (latency_ms={:.1f}, chunks={}, lang={})",
        elapsed_ms,
        len(rag_response.sources),
        rag_response.language,
    )
    return QueryResponse(
        answer=rag_response.answer,
        sources=[_to_source(c) for c in rag_response.sources],
        latency_ms=elapsed_ms,
        language=rag_response.language,
    )


def _to_source(c: RetrievedChunk) -> SourceChunk:
    return SourceChunk(
        id=c.chunk_id,
        text=c.text,
        source=c.source,
        score=c.score,
        metadata=c.metadata,
    )

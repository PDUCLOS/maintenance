"""FastAPI application entry point.

Run with:
    make api        # development (auto-reload)
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 1
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.api.routes import eval as eval_route
from src.api.routes import index as index_route
from src.api.routes import ingest as ingest_route
from src.api.routes import query as query_route
from src.api.schemas import HealthResponse
from src.config import settings
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Verify Chroma reachability at startup. Fail fast if it's not up."""
    logger.info(
        "API starting (log_level={}, chroma={}:{})",
        settings.log_level,
        settings.chroma_host,
        settings.chroma_port,
    )
    try:
        n = VectorStore().count()
        logger.info("Chroma reachable, collection has {} documents", n)
    except Exception as e:
        logger.error("Chroma NOT reachable at startup: {}. Start it with 'make chroma-up'.", e)
        # We don't raise here — the /health endpoint will report the state.
    yield
    logger.info("API shutting down")


app = FastAPI(
    title="Industrial Knowledge Copilot",
    description="Local RAG copilot for industrial maintenance (NASA CMAPSS + technical PDFs).",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — Streamlit UI runs on a different port during dev.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{settings.ui_port}",
        f"http://127.0.0.1:{settings.ui_port}",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(query_route.router)
app.include_router(ingest_route.router)
app.include_router(eval_route.router)
app.include_router(index_route.router)


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    return {
        "name": "Industrial Knowledge Copilot",
        "version": "0.1.0",
        "docs": "/docs",
    }


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness + dependency check."""
    chroma_reachable = False
    collection_count = 0
    try:
        vs = VectorStore()
        collection_count = vs.count()
        chroma_reachable = True
    except Exception as e:
        logger.warning("Health: Chroma unreachable: {}", e)

    # We don't load the MLX model at /health time (it takes 10s).
    # We only check the host is Apple Silicon — the LLM is loaded lazily.
    mlx_ready = settings.is_apple_silicon()
    hardware = "apple_silicon" if mlx_ready else "other"

    overall = "ok" if chroma_reachable and mlx_ready else "degraded"
    return HealthResponse(
        status=overall,
        chroma_reachable=chroma_reachable,
        collection_count=collection_count,
        mlx_ready=mlx_ready,
        hardware=hardware,
    )

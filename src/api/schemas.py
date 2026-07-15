"""Pydantic request/response models for the API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000, description="The user's question.")
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = Field(default=False, description="If true, return NDJSON (one JSON per line).")


class SourceChunk(BaseModel):
    id: str
    text: str
    source: str
    score: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    latency_ms: float
    language: str = Field(
        default="en",
        description="Detected language of the answer (fr or en). Mirror response: same as the question.",
    )


class IngestRequest(BaseModel):
    """Optional knobs. By default the pipeline ingests everything available."""

    include_pdfs: bool = True
    force_rebuild: bool = False


class IngestResponse(BaseModel):
    chunks_indexed: int
    duration_ms: float
    collection: str


class EvalRequest(BaseModel):
    dataset_path: str | None = None  # defaults to settings.eval_dataset_file
    metrics: list[str] = Field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
        ]
    )


class EvalMetricResult(BaseModel):
    name: str
    score: float


class EvalResponse(BaseModel):
    metrics: list[EvalMetricResult]
    n_samples: int
    duration_ms: float
    snapshot_path: str


class HealthResponse(BaseModel):
    status: str
    chroma_reachable: bool
    collection_count: int
    mlx_ready: bool
    hardware: str


class IndexStatsResponse(BaseModel):
    collection_count: int
    source_counts: dict[str, int]
    sample_chunks: list[SourceChunk]

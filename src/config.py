"""Application configuration loaded from environment / .env file.

All settings are validated at import time. Misconfiguration fails fast — no
silent fallbacks, no fake defaults that look like real values.
"""

from __future__ import annotations

import platform
import sys
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Project root (parent of src/) — used to resolve relative paths in .env
PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed application settings.

    Reads from environment variables and a local .env file. All values are
    validated. The first time the LLM or embedding module is loaded, it
    cross-checks the hardware target against the actual platform.
    """

    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Hardware target ----------------------------------------------------
    hardware_target: str = Field(
        default="apple_silicon",
        description="Expected platform. Set to 'apple_silicon' or 'other'.",
    )

    # --- LLM (MLX) ----------------------------------------------------------
    # Qwen2.5-7B-Instruct, not Mistral-7B-Instruct-v0.3: A/B tested on the
    # ReAct agent harness (5 quantitative questions, same 4-bit/7B
    # footprint) — Mistral scored 2/5 clean tool invocations; Qwen2.5
    # scored 5/5. See PLAN.md §8.
    mlx_model_repo: str = Field(
        default="mlx-community/Qwen2.5-7B-Instruct-4bit",
        description="HuggingFace repo id for the MLX-quantized LLM.",
    )
    mlx_model_path: str = Field(
        default="~/.cache/huggingface/hub",
        description="Local HuggingFace cache directory.",
    )
    mlx_max_tokens: int = Field(default=1024, ge=64, le=4096)
    mlx_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    mlx_top_p: float = Field(default=0.9, ge=0.0, le=1.0)

    # --- Embeddings ---------------------------------------------------------
    # NOT an MLX repo despite the field name: src/rag/embeddings.py loads
    # this via sentence-transformers (device="mps"), not MLX. The
    # "-4bit" mlx-community variant ships MLX-quantized weights (packed
    # hidden dim) that a plain BertModel/SentenceTransformer can't load
    # (size mismatch). Use the standard, unquantized checkpoint.
    #
    # bge-m3, not bge-small-en-v1.5: corpus is mixed FR/EN (NTN-SNR docs
    # are French) and users query in French. bge-small-en-v1.5 is
    # English-only — French queries embedded far from English chunks
    # (and vice versa), causing silent retrieval failures with no error,
    # just bad/irrelevant chunks. bge-m3 shares one vector space across
    # languages including French. Changing this requires a full reindex
    # (dimension changed 384 -> 1024, incompatible with the old collection).
    mlx_embed_repo: str = Field(
        default="BAAI/bge-m3",
        description="HuggingFace repo id for the embedding model "
        "(loaded via sentence-transformers, must be a standard/unquantized checkpoint).",
    )
    embed_dim: int = Field(default=1024, description="Embedding vector dimension.")

    # --- ChromaDB -----------------------------------------------------------
    chroma_host: str = Field(default="localhost")
    chroma_port: int = Field(default=8001)
    chroma_collection: str = Field(default="bearings_kb")

    # --- API / UI -----------------------------------------------------------
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    api_log_level: str = Field(default="info")
    ui_port: int = Field(default=8501)

    # --- Paths --------------------------------------------------------------
    data_dir: Path = Field(default=PROJECT_ROOT / "data")
    raw_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw")
    processed_data_dir: Path = Field(default=PROJECT_ROOT / "data" / "processed")
    chunks_file: Path = Field(default=PROJECT_ROOT / "data" / "processed" / "chunks.jsonl")
    eval_dataset_file: Path = Field(
        default=PROJECT_ROOT / "data" / "processed" / "eval_dataset.jsonl"
    )
    pdf_dir: Path = Field(default=PROJECT_ROOT / "data" / "raw" / "pdf")

    # --- RAG ----------------------------------------------------------------
    chunk_size: int = Field(default=500, ge=64, le=2000)
    chunk_overlap: int = Field(default=50, ge=0, le=500)
    retriever_top_k: int = Field(default=5, ge=1, le=50)
    hybrid_search: bool = Field(default=True)
    reranker_enabled: bool = Field(default=True)
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # --- Logging ------------------------------------------------------------
    log_level: str = Field(default="INFO")
    log_format: str = Field(default="json", description="'json' or 'console'.")

    # ----- Helpers ----------------------------------------------------------
    def is_apple_silicon(self) -> bool:
        """Return True if we're running on Apple Silicon (M1/M2/M3/M4/M5)."""
        return platform.system() == "Darwin" and platform.machine() == "arm64"

    def assert_apple_silicon(self) -> None:
        """Raise a clear error if the host is not Apple Silicon.

        MLX only works on Apple Silicon. The rule is: refuse to start with
        a clear error rather than silently fall back to a degraded path.
        """
        if not self.is_apple_silicon():
            raise RuntimeError(
                "MLX requires Apple Silicon (M1/M2/M3/M4/M5). "
                f"Detected: {platform.system()} / {platform.machine()}. "
                "Either run on Apple Silicon hardware or replace the MLX "
                "backend (see docs/architecture.md)."
            )

    def assert_python_version(self) -> None:
        """Raise if Python < 3.12."""
        if sys.version_info < (3, 12):  # noqa: UP036 (runtime guard, not dead code)
            raise RuntimeError(
                f"Python 3.12+ required, found {sys.version_info.major}.{sys.version_info.minor}."
            )


# Singleton — imported everywhere as `from src.config import settings`
settings = Settings()
settings.assert_python_version()

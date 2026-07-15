"""Ingestion pipeline orchestrator.

Runs the full load -> chunk -> embed -> persist flow:

    1. Load CMAPSS readme.txt (text) and per-subset DataFrames
    2. Load PDFs (if any) page by page
    3. Serialize DataFrames to text (per-unit summaries, sensor stats)
    4. Chunk everything with the recursive splitter
    5. Embed with the MLX-backed sentence-transformers model (MPS)
    6. Upsert into ChromaDB (collection from settings.chroma_collection)
    7. Write a JSONL copy of the chunks to data/processed/chunks.jsonl

Usage:
    python -m src.ingestion.pipeline
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import settings
from src.ingestion.chunker import build_chunks
from src.ingestion.cmapss_loader import (
    SUBSETS,
    assert_cmapss_present,
    discover_readme,
    load_train,
)
from src.ingestion.pdf_loader import load_all_pdfs
from src.rag.embeddings import Embedder
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger
from src.utils.timing import timed


@timed
def run() -> None:
    """Execute the full ingestion pipeline. Raises on any error."""
    settings.assert_apple_silicon()
    assert_cmapss_present()

    # 1. Build the raw corpus as (text, source, metadata) tuples
    corpus: list[tuple[str, str, dict[str, str]]] = []

    # 1a. CMAPSS readme
    readme = discover_readme()
    if readme:
        corpus.append((readme.read_text(encoding="utf-8"), "cmapss:readme", {"type": "doc"}))

    # 1b. CMAPSS per-subset structured data (textualised)
    for subset in SUBSETS:
        df = load_train(subset)
        text = _dataframe_to_text(df, subset)
        corpus.append((text, f"cmapss:{subset}", {"type": "dataset", "subset": subset}))

    # 1c. PDFs
    for page in load_all_pdfs():
        corpus.append(
            (page.text, f"pdf:{page.file_name}", {
                "type": "pdf",
                "file_name": page.file_name,
                "page": str(page.page_number),
            })
        )

    logger.info("Corpus: {} raw documents", len(corpus))

    # 2. Chunk
    chunks = build_chunks(corpus)
    logger.info("Produced {} chunks", len(chunks))

    # 3. Persist a JSONL copy (audit trail + cheap re-embed if model changes)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    with settings.chunks_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({
                "chunk_id": c.chunk_id,
                "text": c.text,
                "source": c.source,
                **c.metadata,
            }, ensure_ascii=False) + "\n")
    logger.info("Wrote chunks to {}", settings.chunks_file)

    # 4. Embed + upsert
    embedder = Embedder()
    vectorstore = VectorStore()
    vectorstore.upsert(chunks, embedder.embed([c.text for c in chunks]))
    logger.info("Vector store updated: collection={}", settings.chroma_collection)


def _dataframe_to_text(df, subset: str) -> str:
    """Render a CMAPSS DataFrame as a human-readable markdown text block.

    Includes per-sensor statistics, operating conditions summary, fleet
    size, and per-sensor trend (first 30% vs last 30% of max cycle).
    The LLM will see this text when answering questions about the dataset.

    Note: the trend uses the same definition as `src.eval.dataset` so the
    ground-truth answers in the eval set stay consistent.
    """
    sensor_cols = [c for c in df.columns if c.startswith("sensor_")]
    op_cols = [c for c in df.columns if c.startswith("op_setting_")]

    n_units = int(df["unit_nr"].nunique())
    cycles_per_unit = df.groupby("unit_nr")["time_cycles"].max()
    max_cycle = int(df["time_cycles"].max())

    lines: list[str] = [
        f"# CMAPSS Subset {subset}",
        "",
        f"- Number of engines: {n_units}",
        f"- Total cycles (all engines): {len(df):,}",
        f"- Cycles per engine: min={int(cycles_per_unit.min())}, "
        f"max={int(cycles_per_unit.max())}, mean={cycles_per_unit.mean():.1f}",
        "",
        "## Operating conditions (mean ± std)",
    ]
    for col in op_cols:
        m, s = df[col].mean(), df[col].std()
        lines.append(f"- {col}: {m:.4f} ± {s:.4f}")

    lines.extend(["", "## Sensor statistics (mean ± std)"])
    for col in sensor_cols:
        m, s = df[col].mean(), df[col].std()
        lines.append(f"- {col}: {m:.2f} ± {s:.2f}")

    # Per-sensor trend: compare mean over the first 30% of max cycle vs
    # the last 30% of max cycle. Stable, increase, or decrease.
    lines.extend([
        "",
        "## Sensor trends (first 30% vs last 30% of max cycle)",
    ])
    cutoff = int(max_cycle * 0.3)
    first_mask = df["time_cycles"] <= cutoff
    last_mask = df["time_cycles"] > (max_cycle - cutoff)
    for col in sensor_cols:
        first_mean = df.loc[first_mask, col].mean()
        last_mean = df.loc[last_mask, col].mean()
        if abs(last_mean - first_mean) < 1e-6:
            trend = "stable"
        elif last_mean > first_mean:
            trend = "increases"
        else:
            trend = "decreases"
        lines.append(
            f"- {col}: {trend} (first30_mean={first_mean:.2f}, last30_mean={last_mean:.2f})"
        )

    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    run()

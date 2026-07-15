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
        corpus.append((readme.read_text(encoding="utf-8"), f"cmapss:readme", {"type": "doc"}))

    # 1b. CMAPSS per-subset structured data (textualised)
    from src.ingestion.cmapss_loader import load_train  # local import to keep startup fast

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
    """Render a CMAPSS DataFrame as a human-readable text block.

    Includes per-sensor statistics and operating conditions summary. This
    is what the LLM will see when answering questions about the dataset.
    """
    raise NotImplementedError(
        f"DataFrame -> text: to be implemented in W1 (per-sensor mean/min/max, "
        f"operating condition range, fleet size, for subset {subset})."
    )


if __name__ == "__main__":
    run()

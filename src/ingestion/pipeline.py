"""Ingestion pipeline orchestrator.

Runs the full load -> chunk -> embed -> persist flow for the PDF
catalogue (Schaeffler + SKF):

    1. Load PDFs (if any) page by page
    2. Chunk everything with the recursive splitter
    3. Embed with the sentence-transformers model (MPS)
    4. Upsert into ChromaDB (collection from settings.chroma_collection)
    5. Write a JSONL copy of the chunks to data/processed/chunks.jsonl

Usage:
    python -m src.ingestion.pipeline
"""

from __future__ import annotations

import json

from src.config import settings
from src.ingestion.chunker import build_chunks
from src.ingestion.pdf_loader import load_all_pdfs
from src.rag.embeddings import Embedder
from src.rag.vectorstore import VectorStore
from src.utils.logger import logger
from src.utils.timing import timed


@timed
def run() -> None:
    """Execute the full ingestion pipeline. Raises on any error."""
    settings.assert_apple_silicon()

    # 1. Build the raw corpus as (text, source, metadata) tuples
    corpus: list[tuple[str, str, dict[str, str]]] = []

    # 1a. PDFs (Schaeffler + SKF catalogues)
    for page in load_all_pdfs():
        # Source must include the page number: build_chunks() derives
        # chunk_id from source + a per-source chunk index, so two pages
        # of the same PDF sharing a source would collide on chunk_id
        # (Chroma rejects duplicate IDs within a single upsert).
        corpus.append(
            (
                page.text,
                f"pdf:{page.file_name}:p{page.page_number}",
                {
                    "type": "pdf",
                    "file_name": page.file_name,
                    "page": str(page.page_number),
                },
            )
        )

    if not corpus:
        raise RuntimeError(
            f"No PDF files found in {settings.pdf_dir}. "
            "Add Schaeffler/SKF catalogues there, then re-run."
        )

    logger.info("Corpus: {} raw documents (PDFs only)", len(corpus))

    # 2. Chunk
    chunks = build_chunks(corpus)
    logger.info("Produced {} chunks", len(chunks))

    # 3. Persist a JSONL copy (audit trail + cheap re-embed if model changes)
    settings.processed_data_dir.mkdir(parents=True, exist_ok=True)
    with settings.chunks_file.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(
                json.dumps(
                    {
                        "chunk_id": c.chunk_id,
                        "text": c.text,
                        "source": c.source,
                        **c.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    logger.info("Wrote chunks to {}", settings.chunks_file)

    # 4. Embed + upsert
    embedder = Embedder()
    vectorstore = VectorStore()
    vectorstore.upsert(chunks, embedder.embed([c.text for c in chunks]))
    logger.info("Vector store updated: collection={}", settings.chroma_collection)


__all__ = ["run"]


if __name__ == "__main__":
    run()

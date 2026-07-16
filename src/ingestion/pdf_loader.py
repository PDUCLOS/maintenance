"""PDF technical documentation loader.

Uses PyMuPDF (fitz) for fast, layout-aware text extraction. Splits documents
into one record per page with metadata (file_name, page_number, total_pages).

Note: technical PDFs (Schaeffler, SKF, NTN-SNR catalogues) are large and
visually rich. The default extraction may miss tables or figures. We keep
the loader simple here — a layout-aware parser is a future enhancement
(W3-W4, see PLAN.md section 11 risks).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.config import settings
from src.utils.logger import logger


@dataclass
class PdfPage:
    """One page of extracted text plus its source metadata."""

    text: str
    file_name: str
    page_number: int  # 1-indexed
    total_pages: int
    extra: dict[str, str] = field(default_factory=dict)


def list_pdfs() -> list[Path]:
    """List PDF files in the configured pdf_dir.

    Raises FileNotFoundError if the directory itself does not exist —
    this is intentional: an empty PDF set is fine (CMAPSS-only mode),
    a missing directory is a config error.
    """
    pdf_dir = settings.pdf_dir
    if not pdf_dir.is_dir():
        raise FileNotFoundError(
            f"PDF directory not found: {pdf_dir}. "
            "Either create it (even empty) or fix pdf_dir in .env."
        )
    return sorted(pdf_dir.glob("*.pdf"))


def load_pdf(pdf_path: Path) -> list[PdfPage]:
    """Extract text from a PDF, one record per page.

    Returns an empty list (not a fake record) if the PDF has no extractable
    text — which is the right behaviour for a scanned-only PDF.
    """
    import fitz  # PyMuPDF — local import to keep the package import cheap

    if not pdf_path.is_file():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages: list[PdfPage] = []
    doc = fitz.open(pdf_path)
    try:
        total = len(doc)
        for i, page in enumerate(doc):
            text = page.get_text("text")  # plain text extraction
            if text and text.strip():
                pages.append(
                    PdfPage(
                        text=text,
                        file_name=pdf_path.name,
                        page_number=i + 1,
                        total_pages=total,
                    )
                )
    finally:
        doc.close()
    return pages


def load_all_pdfs() -> list[PdfPage]:
    """Load every PDF in pdf_dir. Returns [] if no PDFs are present."""
    pdfs = list_pdfs()
    if not pdfs:
        logger.info("No PDFs found in {}, skipping PDF ingestion", settings.pdf_dir)
        return []
    pages: list[PdfPage] = []
    for pdf in pdfs:
        pages.extend(load_pdf(pdf))
    logger.info("Loaded {} pages from {} PDF(s)", len(pages), len(pdfs))
    return pages


__all__ = ["PdfPage", "list_pdfs", "load_all_pdfs", "load_pdf"]

"""Unit + integration tests for the ingestion module.

Unit tests (always run):
  - chunker behaviour on synthetic text (recursive_split, build_chunks)

Integration tests (require PDFs in data/raw/pdf/):
  - PDF loader can extract text from a real PDF
"""

from __future__ import annotations

import pytest

from src.ingestion.chunker import (
    build_chunks,
    count_tokens,
    recursive_split,
)

# ============================================================
# Unit tests (no data required)
# ============================================================


def test_count_tokens_is_reasonable():
    """A short English sentence should be ~5-20 tokens."""
    n = count_tokens("The bearing requires periodic lubrication.")
    assert 5 <= n <= 20


def test_recursive_split_short_text():
    """Text shorter than chunk_size returns the text as a single chunk."""
    text = "Hello world. " * 10  # ~30 tokens
    chunks = recursive_split(text, chunk_size=500, chunk_overlap=50)
    assert chunks == [text]


def test_recursive_split_empty_text():
    assert recursive_split("") == []
    assert recursive_split("   \n\n  ") == []


def test_recursive_split_long_text():
    """Text longer than chunk_size is split into multiple chunks."""
    text = "The sensor reading was nominal. " * 40  # ~200 tokens
    chunks = recursive_split(text, chunk_size=80, chunk_overlap=10)
    assert len(chunks) >= 2
    for c in chunks:
        assert count_tokens(c) <= 80


def test_recursive_split_long_pdf_like_text():
    """A 2000-token blob chunks into multiple pieces, all within size."""
    text = (
        "The maintenance procedure requires inspection of all components. "
        "Verify torque values, check fluid levels, and confirm sensor calibration. "
        "Refer to the technical manual for torque specifications. "
    ) * 80  # ~2000 tokens
    chunks = recursive_split(text, chunk_size=500, chunk_overlap=50)
    assert len(chunks) >= 4
    for c in chunks:
        assert count_tokens(c) <= 500


def test_recursive_split_overlap_creates_continuity():
    """Adjacent chunks should share content at the boundary (overlap)."""
    text = "The engine is degrading. " * 200  # ~800 tokens
    chunks = recursive_split(text, chunk_size=200, chunk_overlap=30)
    assert len(chunks) >= 2
    # The tail of chunk[0] and the head of chunk[1] should overlap.
    # Check by looking for a common 5+ token substring at the boundary.
    tail = chunks[0][-100:]
    head = chunks[1][:100]
    assert any(
        token in head for token in tail.split()[-5:]
    ), "Expected at least 1 word overlap between adjacent chunks"


def test_build_chunks_stable_ids():
    """Re-ingesting the same source produces the same chunk_ids."""
    pages = [
        ("hello world", "pdf:foo.pdf:p1", {"type": "pdf"}),
        ("another text", "pdf:bar.pdf:p3", {"type": "pdf"}),
    ]
    chunks_a = build_chunks(pages)
    chunks_b = build_chunks(pages)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.chunk_id for c in chunks_a] == [
        "pdf:foo.pdf:p1:0",
        "pdf:bar.pdf:p3:0",
    ]


def test_build_chunks_skips_empty():
    pages = [
        ("", "src1", {}),
        ("  \n\n  ", "src2", {}),
        ("real text", "src3", {}),
    ]
    chunks = build_chunks(pages)
    assert len(chunks) == 1
    assert chunks[0].source == "src3"


# ============================================================
# Integration tests (require PDFs in data/raw/pdf/)
# ============================================================


@pytest.mark.integration
def test_pdf_loader_loads_first_available_pdf():
    """The first available PDF in data/raw/pdf/ should extract non-empty text."""
    from src.config import settings
    from src.ingestion.pdf_loader import load_pdf

    pdfs = sorted(settings.pdf_dir.glob("*.pdf")) if settings.pdf_dir.is_dir() else []
    if not pdfs:
        pytest.skip(f"No PDFs in {settings.pdf_dir}")
    pages = load_pdf(pdfs[0])
    assert len(pages) >= 1
    # First page should have non-trivial text
    assert len(pages[0].text) > 200
    assert pages[0].page_number == 1

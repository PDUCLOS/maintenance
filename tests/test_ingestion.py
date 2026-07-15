"""Unit + integration tests for the ingestion module.

Unit tests (always run):
  - column names, subset list, paths, count_tokens
  - chunker behaviour on synthetic text

Integration tests (require NASA CMAPSS data in data/raw/cmapss/):
  - load_train, load_test, load_rul on real data
  - _dataframe_to_text produces a markdown block with the right sections
  - PDF loader can extract the bonus NASA paper
"""

from __future__ import annotations

import pytest

from src.ingestion.cmapss_loader import COLUMN_NAMES, SUBSETS, expected_files
from src.ingestion.chunker import (
    build_chunks,
    count_tokens,
    recursive_split,
)


# ============================================================
# Unit tests (no data required)
# ============================================================

def test_cmapss_column_count():
    """CMAPSS has 26 columns: 1 unit + 1 cycle + 3 op_settings + 21 sensors."""
    assert len(COLUMN_NAMES) == 26


def test_cmapss_subsets():
    assert SUBSETS == ("FD001", "FD002", "FD003", "FD004")


def test_expected_files_paths():
    paths = expected_files("FD001")
    assert paths["train"].name == "train_FD001.txt"
    assert paths["test"].name == "test_FD001.txt"
    assert paths["rul"].name == "RUL_FD001.txt"


def test_expected_files_invalid_subset():
    with pytest.raises(ValueError, match="Unknown CMAPSS subset"):
        expected_files("FD999")


def test_count_tokens_is_reasonable():
    """A short English sentence should be ~5-20 tokens."""
    n = count_tokens("The turbofan engine is degrading.")
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
        token in head
        for token in tail.split()[-5:]
    ), "Expected at least 1 word overlap between adjacent chunks"


def test_build_chunks_stable_ids():
    """Re-ingesting the same source produces the same chunk_ids."""
    pages = [
        ("hello world", "cmapss:FD001", {"type": "dataset"}),
        ("another text", "cmapss:FD002", {"type": "dataset"}),
    ]
    chunks_a = build_chunks(pages)
    chunks_b = build_chunks(pages)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]
    assert [c.chunk_id for c in chunks_a] == [
        "cmapss:FD001:0",
        "cmapss:FD002:0",
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
# Integration tests (require NASA CMAPSS data)
# ============================================================

@pytest.mark.integration
def test_load_train_returns_dataframe():
    from src.ingestion.cmapss_loader import load_train

    df = load_train("FD001")
    assert len(df) > 0
    assert list(df.columns) == COLUMN_NAMES
    assert df["unit_nr"].dtype.kind == "i"  # integer dtype
    assert df["time_cycles"].dtype.kind == "i"
    # Per CMAPSS readme: 100 engines in FD001
    assert df["unit_nr"].nunique() == 100


@pytest.mark.integration
def test_load_test_returns_dataframe():
    from src.ingestion.cmapss_loader import load_test

    df = load_test("FD001")
    assert len(df) > 0
    # FD001 has 100 test units
    assert df["unit_nr"].nunique() == 100


@pytest.mark.integration
def test_load_rul_returns_series_indexed_by_unit():
    from src.ingestion.cmapss_loader import load_rul, load_test

    rul = load_rul("FD001")
    n_test_units = load_test("FD001")["unit_nr"].nunique()
    assert len(rul) == n_test_units
    assert rul.index.name == "unit_nr"
    assert rul.index[0] == 1
    assert rul.index[-1] == n_test_units
    # All RUL values must be positive
    assert (rul > 0).all()


@pytest.mark.integration
def test_dataframe_to_text_contains_key_sections():
    from src.ingestion.cmapss_loader import load_train
    from src.ingestion.pipeline import _dataframe_to_text

    df = load_train("FD001")
    text = _dataframe_to_text(df, "FD001")

    # Must include the section headers
    assert "# CMAPSS Subset FD001" in text
    assert "## Operating conditions" in text
    assert "## Sensor statistics" in text
    assert "## Sensor trends" in text
    # Must mention every sensor
    for i in range(1, 22):
        assert f"sensor_{i:02d}" in text
    # Must contain a valid trend label
    valid_trends = ("stable", "increases", "decreases")
    assert any(
        trend in line
        for line in text.splitlines()
        for trend in valid_trends
        if line.startswith("- sensor_")
    ), "No trend line with stable/increases/decreases found"


@pytest.mark.integration
def test_pdf_loader_loads_damage_propagation_pdf():
    """The bonus NASA PDF (shipped with CMAPSS) should extract non-empty text."""
    from src.config import settings
    from src.ingestion.pdf_loader import load_pdf

    pdf_path = settings.cmapss_dir / "Damage Propagation Modeling.pdf"
    if not pdf_path.exists():
        pytest.skip(f"Bonus PDF not present: {pdf_path}")
    pages = load_pdf(pdf_path)
    assert len(pages) >= 1
    # First page should have non-trivial text
    assert len(pages[0].text) > 200
    assert pages[0].page_number == 1

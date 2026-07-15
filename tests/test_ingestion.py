"""Unit tests for the ingestion module.

These tests cover pure logic (chunker, data file detection). They do NOT
require a live CMAPSS download or any external service.
"""

from __future__ import annotations

import pytest

from src.ingestion.cmapss_loader import COLUMN_NAMES, SUBSETS, expected_files
from src.ingestion.chunker import count_tokens


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

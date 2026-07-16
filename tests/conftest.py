"""Shared pytest fixtures.

Two flavours of tests:
  - unit tests (default): no live services required, run anywhere
  - integration tests: marked @pytest.mark.integration, require a live
    ChromaDB on :8001 and Apple Silicon + downloaded MLX models

The `addopts -m not integration` line in pyproject.toml means integration
tests are SKIPPED by default. Run them explicitly with:
    make test-integration        # locally
"""

from __future__ import annotations

import pytest

from src.config import settings


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests on non-Apple-Silicon or when PDFs are missing."""
    skip_msilicon = pytest.mark.skip(reason="Apple Silicon required (MLX)")
    skip_pdfs = pytest.mark.skip(reason="No PDFs found in data/raw/pdf/")
    for item in items:
        if "integration" in item.keywords:
            if not settings.is_apple_silicon():
                item.add_marker(skip_msilicon)
            if not settings.pdf_dir.is_dir() or not any(settings.pdf_dir.glob("*.pdf")):
                item.add_marker(skip_pdfs)

"""Shared pytest fixtures.

Two flavours of tests:
  - unit tests (default): no live services required, run anywhere
  - integration tests: marked @pytest.mark.integration, require a live
    ChromaDB on :8001 and Apple Silicon + downloaded MLX models

The `addopts -m not integration` line in pyproject.toml means integration
tests are SKIPPED by default. Run them explicitly with:
    make test-integration        # locally
    pytest -m integration -v     # CI on a self-hosted runner
"""

from __future__ import annotations

import pytest

from src.config import settings


def pytest_collection_modifyitems(config, items):
    """Auto-skip integration tests on non-Apple-Silicon or when CMAPSS is missing."""
    skip_msilicon = pytest.mark.skip(reason="Apple Silicon required (MLX)")
    skip_cmapss = pytest.mark.skip(reason="NASA CMAPSS data not found in data/raw/cmapss/")
    for item in items:
        if "integration" in item.keywords:
            if not settings.is_apple_silicon():
                item.add_marker(skip_msilicon)
            try:
                from src.ingestion.cmapss_loader import assert_cmapss_present
                assert_cmapss_present()
            except FileNotFoundError:
                item.add_marker(skip_cmapss)

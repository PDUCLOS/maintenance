"""Integration tests for the FastAPI endpoints.

Requires a running API on :8000 and ChromaDB on :8001 with ingested data.
Run with:
    make api &
    make test-integration
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_health():
    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["hardware"] in ("apple_silicon", "other")
        assert "collection_count" in body


@pytest.mark.integration
def test_query_endpoint():
    from src.api.main import app

    with TestClient(app) as client:
        r = client.post("/query", json={"question": "What is CMAPSS?", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "latency_ms" in body
        assert isinstance(body["sources"], list)


@pytest.mark.integration
def test_query_validation():
    """Empty question must be rejected by the Pydantic schema."""
    from src.api.main import app

    with TestClient(app) as client:
        r = client.post("/query", json={"question": "", "top_k": 5})
        assert r.status_code == 422

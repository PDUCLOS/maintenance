"""Tests for the FastAPI endpoints.

Most are @pytest.mark.integration (need live Chroma + ingested data).
The timeout test is a unit test that mocks chain.query() to be slow.
"""

from __future__ import annotations

import time

import pytest

# ===========================================================================
# Unit tests (no live services)
# ===========================================================================


class TestQueryTimeout:
    """The /query endpoint must return 504 if chain.query() exceeds
    the configured timeout. This is a defensive test — it prevents
    the API from hanging forever if MLX crashes or hangs."""

    def test_timeout_returns_504(self, monkeypatch):
        from fastapi.testclient import TestClient

        # Set a tiny timeout so the test is fast
        from src.config import settings

        monkeypatch.setattr(settings, "query_timeout_seconds", 0.2)

        # Mock chain.query() to take longer than the timeout
        from src.rag.chain import RAGChain

        class FakeResponse:
            def __init__(self):
                self.answer = "late answer"
                self.sources = []
                self.language = "en"

        def slow_query(question, top_k=5):
            time.sleep(2)  # 2s, way over the 0.2s timeout
            return FakeResponse()

        # RAGChain.get() returns a singleton; replace its .query method
        real_get = RAGChain.get
        monkeypatch.setattr(
            RAGChain,
            "get",
            classmethod(lambda cls: type("FakeChain", (), {"query": staticmethod(slow_query)})()),
        )

        from src.api.main import app

        with TestClient(app) as client:
            r = client.post("/query", json={"question": "test?", "top_k": 3})
            assert r.status_code == 504
            body = r.json()
            assert body["error"] == "query_timeout"
            assert "did not return" in body["message"]
            assert body["elapsed_ms"] > 200  # at least 200ms (the timeout)

        # Restore for other tests
        monkeypatch.setattr(RAGChain, "get", real_get)

    def test_quick_query_returns_200(self, monkeypatch):
        """Sanity check: a fast chain.query() still returns 200 (the
        timeout doesn't accidentally always trigger)."""
        from fastapi.testclient import TestClient

        from src.config import settings

        monkeypatch.setattr(settings, "query_timeout_seconds", 5)

        from src.rag.chain import RAGChain

        class FakeResponse:
            def __init__(self):
                self.answer = "fast answer"
                self.sources = []
                self.language = "en"

        def quick_query(question, top_k=5):
            return FakeResponse()

        real_get = RAGChain.get
        monkeypatch.setattr(
            RAGChain,
            "get",
            classmethod(lambda cls: type("FakeChain", (), {"query": staticmethod(quick_query)})()),
        )

        from src.api.main import app

        with TestClient(app) as client:
            r = client.post("/query", json={"question": "test?", "top_k": 3})
            assert r.status_code == 200
            assert r.json()["answer"] == "fast answer"

        monkeypatch.setattr(RAGChain, "get", real_get)

    def test_internal_error_returns_500(self, monkeypatch):
        """A chain.query() exception must return 500 with a clean body,
        not crash the server with a 500 traceback."""
        from fastapi.testclient import TestClient

        from src.rag.chain import RAGChain

        def broken_query(question, top_k=5):
            raise ValueError("test boom")

        real_get = RAGChain.get
        monkeypatch.setattr(
            RAGChain,
            "get",
            classmethod(lambda cls: type("FakeChain", (), {"query": staticmethod(broken_query)})()),
        )

        from src.api.main import app

        with TestClient(app) as client:
            r = client.post("/query", json={"question": "test?", "top_k": 3})
            assert r.status_code == 500
            body = r.json()
            assert body["error"] == "internal_error"
            assert "test boom" in body["message"]

        monkeypatch.setattr(RAGChain, "get", real_get)


# ===========================================================================
# Integration tests (require live services)
# ===========================================================================


@pytest.mark.integration
def test_health():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["hardware"] in ("apple_silicon", "other")
        assert "collection_count" in body


@pytest.mark.integration
def test_query_endpoint():
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.post("/query", json={"question": "What is a rolling bearing?", "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert "answer" in body
        assert "sources" in body
        assert "latency_ms" in body
        assert isinstance(body["sources"], list)


@pytest.mark.integration
def test_query_validation():
    """Empty question must be rejected by the Pydantic schema."""
    from fastapi.testclient import TestClient

    from src.api.main import app

    with TestClient(app) as client:
        r = client.post("/query", json={"question": "", "top_k": 5})
        assert r.status_code == 422

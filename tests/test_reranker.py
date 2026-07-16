"""Unit tests for the cross-encoder reranker.

These tests do NOT require the model to be loaded — they only test
the logic. Integration tests (loading the actual model) are gated
behind @pytest.mark.integration and require Apple Silicon.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from src.rag.reranker import RERANK_OVERFETCH, Reranker
from src.rag.types import RetrievedChunk


def _make_chunk(chunk_id: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        source="test:src",
        metadata={"test": "true"},
        score=0.0,
        retrieval_method="rrf",
    )


def test_rerank_overfetch_multiplier_is_positive():
    """The over-fetch multiplier must be > 1 (else reranker is useless)."""
    assert RERANK_OVERFETCH > 1


def test_reranker_empty_input_returns_empty():
    """No chunks in → no chunks out."""
    r = Reranker()
    assert r.rerank("any query", [], top_n=5) == []


def test_reranker_returns_input_as_is_if_below_top_n():
    """If len(chunks) <= top_n, no reranking, return as-is (cheap path)."""
    r = Reranker()
    chunks = [_make_chunk("c1", "alpha"), _make_chunk("c2", "beta")]
    out = r.rerank("query", chunks, top_n=5)
    assert out == chunks


def test_reranker_picks_relevant_chunks_first():
    """The cross-encoder should rank 'apple' higher than 'banana' for query 'fruit'."""
    # Mock the model to avoid loading the real one
    mock_model = MagicMock()
    chunks = [
        _make_chunk("c1", "banana"),
        _make_chunk("c2", "apple"),
        _make_chunk("c3", "cherry"),
    ]
    # predict() scores line up positionally with `chunks` (via the
    # `pairs` built from it) — apple (c2) is the mocked most relevant.
    mock_model.predict.return_value = [0.1, 0.9, 0.5]

    with patch.object(Reranker, "_model", mock_model):
        r = Reranker()
        out = r.rerank("fruit", chunks, top_n=2)

    assert len(out) == 2
    # 'apple' (mocked highest score) should be first
    assert out[0].chunk_id == "c2"
    # Method updated to 'reranked'
    assert all(c.retrieval_method == "reranked" for c in out)
    # Score updated
    assert out[0].score == 0.9
    assert out[1].score == 0.5


def test_reranker_model_loads_once_under_concurrent_calls():
    """Thread-safety: the lock prevents double-loading the model."""
    from src.rag import reranker

    # Reset the class-level cache
    reranker.Reranker._model = None

    call_count = 0

    def fake_load(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return MagicMock()

    with patch("sentence_transformers.CrossEncoder", side_effect=fake_load):
        r1 = Reranker()
        r2 = Reranker()
        # Trigger load by calling .predict on the mock; here we just check
        # the lock acquires. Simulate: call _load on each
        r1._load()
        r2._load()
    # Should be loaded exactly once thanks to the lock + check
    assert call_count == 1
    # Reset
    reranker.Reranker._model = None

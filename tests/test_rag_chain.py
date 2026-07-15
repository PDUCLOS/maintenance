"""Integration tests for the RAG chain.

These tests require:
  - Apple Silicon (MLX)
  - Downloaded MLX model in the HF cache (run `make pull-models`)
  - Ingested data in ChromaDB (run `make ingest`)
  - ChromaDB reachable on :8001 (run `make chroma-up`)

They are marked @pytest.mark.integration and skipped by default.
"""

from __future__ import annotations

import pytest


@pytest.mark.integration
def test_rag_chain_smoke():
    """A factual question should return a non-empty answer with sources."""
    from src.rag.chain import RAGChain

    chain = RAGChain.get()
    response = chain.query("How many engines are in the FD001 training set?")
    assert response.answer.strip()
    assert len(response.sources) > 0
    # The answer should mention a number (engine count)
    assert any(c.isdigit() for c in response.answer)


@pytest.mark.integration
def test_rag_chain_refuses_out_of_scope():
    """Out-of-scope questions should NOT get a confident answer."""
    from src.rag.chain import RAGChain

    chain = RAGChain.get()
    response = chain.query("What is the phone number of NASA support?")
    assert "don't know" in response.answer.lower() or "je ne sais pas" in response.answer.lower()

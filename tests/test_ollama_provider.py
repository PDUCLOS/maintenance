"""Unit tests for the portable backends (Ollama LLM/embeddings, embedded Chroma).

No live Ollama is needed: httpx calls are monkeypatched with canned
responses shaped like Ollama's /api/chat and /api/embed payloads.
"""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import Settings, settings


class _FakeResponse:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return self._payload


class TestOllamaChatModel:
    def test_generate_sends_messages_and_returns_usage(self, monkeypatch):
        import httpx

        from src.rag.llm import OllamaChatModel

        captured: dict = {}

        def fake_post(url, json, timeout):
            captured["url"] = url
            captured["body"] = json
            return _FakeResponse(
                {
                    "message": {"role": "assistant", "content": " Answer [1]. STOP tail"},
                    "prompt_eval_count": 120,
                    "eval_count": 30,
                }
            )

        monkeypatch.setattr(httpx, "post", fake_post)
        llm = OllamaChatModel(model_name="qwen3.5:9b", base_url="http://ollama:11434")
        msg = llm.invoke([SystemMessage(content="sys"), HumanMessage(content="q?")], stop=["STOP"])

        assert captured["url"] == "http://ollama:11434/api/chat"
        assert captured["body"]["model"] == "qwen3.5:9b"
        assert captured["body"]["think"] is False
        assert captured["body"]["options"]["stop"] == ["STOP"]
        assert [m["role"] for m in captured["body"]["messages"]] == ["system", "user"]
        assert msg.content == "Answer [1]."
        assert msg.usage_metadata == {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150}

    def test_stream_yields_chunks_until_done(self, monkeypatch):
        import httpx

        from src.rag.llm import OllamaChatModel

        lines = [
            json.dumps({"message": {"content": "Hel"}, "done": False}),
            "",
            json.dumps({"message": {"content": "lo"}, "done": False}),
            json.dumps({"message": {"content": ""}, "done": True, "eval_count": 2}),
        ]

        class _Stream:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                return None

            def iter_lines(self):
                return iter(lines)

        monkeypatch.setattr(httpx, "stream", lambda *a, **k: _Stream())
        llm = OllamaChatModel()
        assert "".join(c.content for c in llm.stream("hi")) == "Hello"


class TestBackendSelection:
    def test_factory_returns_ollama_model(self, monkeypatch):
        from src.rag import llm as llm_mod

        monkeypatch.setattr(settings, "llm_provider", "ollama")
        assert isinstance(llm_mod.get_chat_model(), llm_mod.OllamaChatModel)

    def test_mlx_off_apple_silicon_still_fails_fast(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "mlx")
        monkeypatch.setattr(Settings, "is_apple_silicon", lambda self: False)
        with pytest.raises(RuntimeError, match="Apple Silicon"):
            settings.assert_backend_supported()

    def test_unknown_provider_rejected(self, monkeypatch):
        monkeypatch.setattr(settings, "llm_provider", "openai")
        with pytest.raises(RuntimeError, match="LLM_PROVIDER"):
            settings.assert_backend_supported()

    def test_explicit_torch_device_is_kept(self, monkeypatch):
        monkeypatch.setattr(settings, "torch_device", "cpu")
        assert settings.resolve_torch_device() == "cpu"


class TestOllamaEmbeddings:
    def test_vectors_are_normalised_and_batched(self, monkeypatch):
        import httpx

        from src.rag.embeddings import Embedder

        calls: list[list[str]] = []

        class _Client:
            def __init__(self, *a, **k):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def post(self, url, json):
                calls.append(json["input"])
                return _FakeResponse({"embeddings": [[3.0, 4.0] for _ in json["input"]]})

        monkeypatch.setattr(httpx, "Client", _Client)
        monkeypatch.setattr(settings, "embed_provider", "ollama")
        emb = Embedder()
        emb._OLLAMA_BATCH = 2
        vecs = emb.embed(["a", "b", "c"])

        assert calls == [["a", "b"], ["c"]]
        assert vecs[0] == pytest.approx([0.6, 0.8])
        assert len(vecs) == 3
        assert emb.embed([]) == []


class TestEmbeddedChroma:
    def test_persistent_mode_round_trip(self, monkeypatch, tmp_path):
        from src.ingestion.chunker import Chunk
        from src.rag.vectorstore import VectorStore

        monkeypatch.setattr(settings, "chroma_mode", "persistent")
        monkeypatch.setattr(settings, "chroma_persist_dir", tmp_path / "chroma")
        monkeypatch.setattr(settings, "chroma_collection", "unit_test_kb")
        vs = VectorStore()
        vs.upsert(
            [
                Chunk(chunk_id="c1", text="deep groove ball bearing", source="s1", metadata={}),
                Chunk(chunk_id="c2", text="tapered roller bearing", source="s2", metadata={}),
            ],
            [[1.0, 0.0], [0.0, 1.0]],
        )
        hits = vs.query([1.0, 0.0], top_k=1)
        assert vs.count() == 2
        assert hits[0]["id"] == "c1"
        assert hits[0]["metadata"]["source"] == "s1"

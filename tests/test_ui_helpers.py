"""Unit tests for the Streamlit UI helpers (api_client + tabs).

We don't test the tab renderers themselves (they call `st.*` all
over the place and Streamlit is not friendly to headless tests).
What we *do* test is the part that's easy to break and easy to
verify: the HTTP client (api_get / api_post) and the small pure
helpers in the tab modules (brand inference, metric coloring,
JSON loading).

For api_client, we monkeypatch httpx.get / httpx.post so no
network is touched. For the tab helpers, they're pure functions
extracted from the Streamlit-heavy tab bodies and tested directly.
"""

from __future__ import annotations

import json
from pathlib import Path

# ===========================================================================
# src.ui.api_client
# ===========================================================================


class TestApiGet:
    """httpx.get is monkeypatched so no real network is touched."""

    def test_returns_dict_on_200(self, monkeypatch):
        from src.ui import api_client

        def fake_get(url, timeout=None):
            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"status": "ok", "n": 42}

            return _Resp()

        monkeypatch.setattr(api_client.httpx, "get", fake_get)
        result = api_client.api_get("/health", timeout=1.0)
        assert result == {"status": "ok", "n": 42}

    def test_returns_none_on_connection_error(self, monkeypatch):
        from src.ui import api_client

        def fake_get(url, timeout=None):
            raise ConnectionError("nope")

        monkeypatch.setattr(api_client.httpx, "get", fake_get)
        result = api_client.api_get("/health", timeout=1.0)
        assert result is None

    def test_returns_none_on_timeout(self, monkeypatch):
        from src.ui import api_client

        def fake_get(url, timeout=None):
            import httpx as _httpx

            raise _httpx.ConnectTimeout("timed out", request=None)

        monkeypatch.setattr(api_client.httpx, "get", fake_get)
        result = api_client.api_get("/health", timeout=1.0)
        assert result is None

    def test_returns_none_on_http_error(self, monkeypatch):
        """A 500 from the API should return None (caller checks)."""
        import httpx as _httpx

        from src.ui import api_client

        def fake_get(url, timeout=None):
            class _Resp:
                status_code = 500

                def raise_for_status(self):
                    raise _httpx.HTTPStatusError("500 Server Error", request=None, response=self)

            return _Resp()

        monkeypatch.setattr(api_client.httpx, "get", fake_get)
        result = api_client.api_get("/health", timeout=1.0)
        assert result is None


class TestApiPost:
    """httpx.post is monkeypatched so no real network is touched."""

    def test_returns_dict_on_200(self, monkeypatch):
        from src.ui import api_client

        captured = {}

        def fake_post(url, json=None, timeout=None):
            captured["url"] = url
            captured["payload"] = json

            class _Resp:
                status_code = 200

                def raise_for_status(self):
                    pass

                def json(self):
                    return {"answer": "hello", "sources": []}

            return _Resp()

        monkeypatch.setattr(api_client.httpx, "post", fake_post)
        result = api_client.api_post("/query", {"question": "hi"}, timeout=1.0)
        assert result == {"answer": "hello", "sources": []}
        assert captured["url"].endswith("/query")
        assert captured["payload"] == {"question": "hi"}

    def test_returns_none_on_connection_error(self, monkeypatch):
        from src.ui import api_client

        def fake_post(url, json=None, timeout=None):
            raise ConnectionError("refused")

        monkeypatch.setattr(api_client.httpx, "post", fake_post)
        result = api_client.api_post("/query", {"question": "hi"}, timeout=1.0)
        assert result is None


class TestApiBaseUrl:
    """The base URL must be 'localhost:<port>' — NOT 0.0.0.0 (the bind
    interface). Using 0.0.0.0 in the client base URL would resolve to
    the wrong host on some macOS setups."""

    def test_base_url_uses_localhost(self):
        from src.ui import api_client

        # The module-level constant is set at import time
        assert "localhost" in api_client._BASE_URL
        assert "0.0.0.0" not in api_client._BASE_URL


# ===========================================================================
# src.ui.tabs.inventory — pure helpers
# ===========================================================================


class TestInferBrand:
    """Brand inference from PDF filename. Catches typos and
    regressions if someone adds a new manufacturer."""

    def test_schaeffler(self):
        from src.ui.tabs.inventory import _infer_brand

        assert _infer_brand("schaeffler_catalog_2023.pdf") == "Schaeffler"
        assert _infer_brand("Schaeffler_Bearing_Catalogue.pdf") == "Schaeffler"

    def test_fag_and_ina_are_schaeffler(self):
        """FAG and INA are Schaeffler brands."""
        from src.ui.tabs.inventory import _infer_brand

        assert _infer_brand("fag_rolling_bearings.pdf") == "Schaeffler"
        assert _infer_brand("INA_needle_roller.pdf") == "Schaeffler"

    def test_skf(self):
        from src.ui.tabs.inventory import _infer_brand

        assert _infer_brand("skf_general_catalogue.pdf") == "SKF"
        assert _infer_brand("SKF_Bearing_Catalogue.pdf") == "SKF"

    def test_ntn_snr(self):
        from src.ui.tabs.inventory import _infer_brand

        assert _infer_brand("ntn-snr_bearings.pdf") == "NTN-SNR"
        assert _infer_brand("NTN_SNR_Catalogue.pdf") == "NTN-SNR"
        assert _infer_brand("snr_bearings.pdf") == "NTN-SNR"

    def test_unknown_brand_returns_question_mark(self):
        from src.ui.tabs.inventory import _infer_brand

        assert _infer_brand("random_document.pdf") == "?"

    def test_case_insensitive(self):
        from src.ui.tabs.inventory import _infer_brand

        assert _infer_brand("SKF.pdf") == "SKF"
        assert _infer_brand("skf.pdf") == "SKF"
        assert _infer_brand("Skf.pdf") == "SKF"


# ===========================================================================
# src.ui.tabs.ragas — pure helpers
# ===========================================================================


class TestMetricColor:
    """Color-coded metric badges. Thresholds documented in
    docs/evaluation.md — RAGAS targets > 0.75 for the main metrics."""

    def test_green_above_75(self):
        from src.ui.tabs.ragas import _metric_color

        assert _metric_color(0.80) == "🟢"
        assert _metric_color(0.95) == "🟢"
        assert _metric_color(1.00) == "🟢"

    def test_yellow_between_50_and_75(self):
        from src.ui.tabs.ragas import _metric_color

        assert _metric_color(0.50) == "🟡"
        assert _metric_color(0.65) == "🟡"
        assert _metric_color(0.749) == "🟡"

    def test_red_below_50(self):
        from src.ui.tabs.ragas import _metric_color

        assert _metric_color(0.0) == "🔴"
        assert _metric_color(0.30) == "🔴"
        assert _metric_color(0.499) == "🔴"

    def test_exact_threshold_75_is_green(self):
        """The threshold is `>= 0.75` per the original Streamlit code."""
        from src.ui.tabs.ragas import _metric_color

        assert _metric_color(0.75) == "🟢"


class TestLoadSnapshot:
    """Load + parse a snapshot JSON. Should never raise on valid
    eval_*.json files (the eval runner is the producer)."""

    def test_round_trip(self, tmp_path: Path):
        from src.ui.tabs.ragas import _load_snapshot

        snap = tmp_path / "eval_2026-07-15T12-00-00.json"
        payload = {
            "timestamp_utc": "2026-07-15T12:00:00Z",
            "n_samples": 25,
            "metrics": [
                {"name": "faithfulness", "score": 0.82},
                {"name": "answer_relevancy", "score": 0.74},
            ],
        }
        snap.write_text(json.dumps(payload), encoding="utf-8")
        loaded = _load_snapshot(snap)
        assert loaded == payload
        assert loaded["n_samples"] == 25
        assert len(loaded["metrics"]) == 2

"""Tests for the Prometheus-style /metrics endpoint.

We test both the collector (in-process state) and the endpoint
(integration with FastAPI's TestClient). No live services needed.
"""

from __future__ import annotations

import pytest

# ===========================================================================
# MetricsCollector — pure logic
# ===========================================================================


class TestMetricsCollector:
    def setup_method(self):
        from src.api.metrics import reset_collector

        reset_collector()

    def test_empty_render_has_headers_but_no_samples(self):
        from src.api.metrics import get_collector

        out = get_collector().render()
        assert "# TYPE api_requests_total counter" in out
        assert "# TYPE api_request_duration_seconds histogram" in out
        # No samples yet — just the headers
        assert "api_requests_total{" not in out

    def test_observe_increments_counter(self):
        from src.api.metrics import get_collector

        c = get_collector()
        c.observe("/health", "GET", 200, 0.005)
        c.observe("/health", "GET", 200, 0.003)
        c.observe("/v1/query", "POST", 200, 1.234)
        out = c.render()
        assert 'api_requests_total{route="/health",method="GET",status="200"} 2' in out
        assert 'api_requests_total{route="/v1/query",method="POST",status="200"} 1' in out

    def test_histogram_buckets_cumulative(self):
        """Each bucket is a cumulative count of observations <= its upper bound."""
        from src.api.metrics import get_collector

        c = get_collector()
        # 3 obs at 0.005s, 2 obs at 0.5s, 1 obs at 5s
        for _ in range(3):
            c.observe("/x", "GET", 200, 0.005)
        for _ in range(2):
            c.observe("/x", "GET", 200, 0.5)
        c.observe("/x", "GET", 200, 5.0)
        out = c.render()

        # Cumulative: bucket 0.01 catches the 3 short ones; bucket 0.5
        # catches those + the 2 medium ones (total 5); bucket 5 catches
        # those + the 1 long one (total 6, because 5.0 <= 5.0).
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="0.01"} 3' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="0.05"} 3' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="0.1"} 3' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="0.5"} 5' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="1.0"} 5' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="2.0"} 5' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="5.0"} 6' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="10.0"} 6' in out
        assert 'api_request_duration_seconds_bucket{route="/x",method="GET",le="+Inf"} 6' in out

    def test_histogram_sum_and_count(self):
        from src.api.metrics import get_collector

        c = get_collector()
        c.observe("/x", "GET", 200, 0.1)
        c.observe("/x", "GET", 200, 0.2)
        c.observe("/x", "GET", 200, 0.05)
        out = c.render()
        assert 'api_request_duration_seconds_count{route="/x",method="GET"} 3' in out
        assert 'api_request_duration_seconds_sum{route="/x",method="GET"} 0.350000' in out

    def test_output_is_valid_prometheus_text_format(self):
        """The exposition format is line-oriented: each line is one
        metric sample or a HELP/TYPE comment. Prometheus parsers
        tolerate either metric_name OR metric_name{labels} per line."""
        from src.api.metrics import get_collector

        c = get_collector()
        c.observe("/health", "GET", 200, 0.01)
        c.observe("/health", "GET", 200, 0.01)
        c.observe("/v1/query", "POST", 504, 60.0)
        out = c.render()

        lines = out.strip().split("\n")
        # All non-empty lines are either comments (#) or metric samples
        for line in lines:
            assert (
                line.startswith("#") or "{" in line or "}" in line or line.startswith("api_")
            ), f"unexpected line: {line!r}"

    def test_process_singleton(self):
        from src.api.metrics import get_collector

        c1 = get_collector()
        c2 = get_collector()
        assert c1 is c2

    def test_reset_clears_state(self):
        from src.api.metrics import get_collector, reset_collector

        c = get_collector()
        c.observe("/x", "GET", 200, 0.01)
        assert "api_requests_total{" in c.render()
        reset_collector()
        # New collector, fresh state
        out = get_collector().render()
        assert "api_requests_total{" not in out


# ===========================================================================
# /metrics endpoint — integration with FastAPI TestClient
# ===========================================================================


@pytest.fixture
def client():
    """A FastAPI app with a couple of throwaway routes + /metrics."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from src.api.metrics import install_metrics, reset_collector

    app = FastAPI()
    install_metrics(app)

    @app.get("/health")
    def _health():
        return {"ok": True}

    @app.post("/query")
    def _query():
        return {"answer": "x"}

    reset_collector()
    with TestClient(app) as c:
        yield c


class TestMetricsEndpoint:
    def test_metrics_endpoint_returns_text_plain(self, client):
        r = client.get("/metrics")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/plain")
        # Prometheus exposition format MIME
        assert "version=0.0.4" in r.headers["content-type"]

    def test_get_health_increments_counter(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        r = client.get("/metrics")
        body = r.text
        assert 'api_requests_total{route="/health",method="GET",status="200"} 1' in body
        # Histogram count matches the request count
        assert 'api_request_duration_seconds_count{route="/health",method="GET"} 1' in body

    def test_post_query_separate_label_set(self, client):
        client.post("/query", json={"q": "hi"})
        body = client.get("/metrics").text
        assert 'api_requests_total{route="/query",method="POST",status="200"} 1' in body
        # /health was never hit in this test, so the label set should
        # not be emitted (Prometheus convention: no observations = no
        # sample, not a zero-valued sample).
        assert 'api_requests_total{route="/health"' not in body

    def test_404_routes_use_raw_path(self, client):
        """If a request hits a path that's not a registered route
        (404), we still record metrics but use the raw path. In
        practice this is fine — it only happens during a misconfig."""
        client.get("/this-route-does-not-exist")
        body = client.get("/metrics").text
        assert (
            'api_requests_total{route="/this-route-does-not-exist",method="GET",status="404"} 1'
            in body
        )

    def test_500_status_recorded_on_exception(self, client):
        """If a handler raises, the middleware still records the request
        with status 500 (the default we set in the finally block)."""
        from fastapi import FastAPI, Request
        from fastapi.responses import JSONResponse
        from fastapi.testclient import TestClient

        from src.api.metrics import install_metrics, reset_collector

        app = FastAPI()
        install_metrics(app)

        @app.exception_handler(RuntimeError)
        async def _on_runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
            return JSONResponse(status_code=500, content={"error": str(exc)})

        @app.get("/boom")
        def _boom():
            raise RuntimeError("kaboom")

        reset_collector()
        with TestClient(app) as c:
            r = c.get("/boom")
            assert r.status_code == 500
            body = c.get("/metrics").text
            assert 'api_requests_total{route="/boom",method="GET",status="500"} 1' in body

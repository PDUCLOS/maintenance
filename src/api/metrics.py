"""Lightweight Prometheus-style metrics for the FastAPI app.

We don't pull in `prometheus_client` (heavy dep) for what is essentially
3 lines of state: a counter, a histogram bucket map, and a small
text-exposition format printer. The format we emit is the Prometheus
text exposition v0.0.4 — same one a real `prometheus_client` instance
would emit, so any scraper (Prometheus, VictoriaMetrics, Grafana Agent,
Mimir) reads it as-is.

What's tracked:
  - `api_requests_total{route, method, status}` — counter of HTTP
    requests per route+method+status code.
  - `api_request_duration_seconds_bucket{route, method, le}` — histogram
    of request latencies with the bucket upper bounds 0.01, 0.05, 0.1,
    0.5, 1, 2, 5, 10, 30 s. The "+Inf" bucket is the total.
  - `api_request_duration_seconds_sum{route, method}` — total seconds
    observed.
  - `api_request_duration_seconds_count{route, method}` — total number
    of latency observations.

The collector is a process-singleton (one FastAPI process = one
collector). Multi-worker deployments (e.g. uvicorn --workers 4) would
each have their own copy, which is normal for client-side metrics —
the scraper aggregates per-instance via the `instance` label.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse

# Histogram bucket upper bounds (in seconds). Chosen to cover
# everything from cached health checks (1ms) up to slow LLM
# generations (10-30s). The +Inf bucket is implicit.
_BUCKETS: tuple[float, ...] = (0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0)


class MetricsCollector:
    """In-memory counters + histograms. Thread-safe (one lock)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # request_count[(route, method, status)] = int
        self._request_count: dict[tuple[str, str, str], int] = defaultdict(int)
        # hist_buckets[(route, method)][bucket_le] = int
        self._hist_buckets: dict[tuple[str, str], dict[float, int]] = defaultdict(
            lambda: {b: 0 for b in _BUCKETS}
        )
        # hist_count[(route, method)] = int
        self._hist_count: dict[tuple[str, str], int] = defaultdict(int)
        # hist_sum[(route, method)] = float
        self._hist_sum: dict[tuple[str, str], float] = defaultdict(float)

    def observe(self, route: str, method: str, status: int, duration_s: float) -> None:
        """Called from the middleware on every request."""
        key = (route, method)
        status_str = str(status)
        with self._lock:
            self._request_count[(route, method, status_str)] += 1
            self._hist_count[key] += 1
            self._hist_sum[key] += duration_s
            buckets = self._hist_buckets[key]
            for b in _BUCKETS:
                if duration_s <= b:
                    buckets[b] += 1

    def render(self) -> str:
        """Render in Prometheus text exposition format (v0.0.4)."""
        lines: list[str] = []
        # Header
        lines.append("# HELP api_requests_total Total HTTP requests handled by the API.")
        lines.append("# TYPE api_requests_total counter")
        with self._lock:
            # api_requests_total
            for (route, method, status), count in sorted(self._request_count.items()):
                lines.append(
                    f'api_requests_total{{route="{route}",method="{method}",status="{status}"}} {count}'
                )
            # api_request_duration_seconds
            lines.append("# HELP api_request_duration_seconds HTTP request latency in seconds.")
            lines.append("# TYPE api_request_duration_seconds histogram")
            for key in sorted(self._hist_count):
                route, method = key
                buckets = self._hist_buckets[key]
                cumulative = 0
                for b in _BUCKETS:
                    cumulative = buckets[b]
                    lines.append(
                        f'api_request_duration_seconds_bucket{{route="{route}",method="{method}",le="{b}"}} {cumulative}'
                    )
                # +Inf bucket = total count
                lines.append(
                    f'api_request_duration_seconds_bucket{{route="{route}",method="{method}",le="+Inf"}} {self._hist_count[key]}'
                )
                lines.append(
                    f'api_request_duration_seconds_sum{{route="{route}",method="{method}"}} {self._hist_sum[key]:.6f}'
                )
                lines.append(
                    f'api_request_duration_seconds_count{{route="{route}",method="{method}"}} {self._hist_count[key]}'
                )
        return "\n".join(lines) + "\n"


# Process-singleton (created lazily on first call to get_collector()).
_collector: MetricsCollector | None = None
_collector_lock = threading.Lock()


def get_collector() -> MetricsCollector:
    global _collector
    if _collector is None:
        with _collector_lock:
            if _collector is None:
                _collector = MetricsCollector()
    return _collector


def reset_collector() -> None:
    """For tests: clear all metrics."""
    global _collector
    with _collector_lock:
        _collector = None


def _route_template(request: Request) -> str:
    """Return the FastAPI route template (e.g. '/v1/query'), or the
    raw path if the request didn't match a registered route.

    Using the raw path would explode cardinality — every unique chunk_id
    in a /query response would be a separate label. The route template
    keeps cardinality bounded to 'number of routes' (currently <10).
    """
    route = request.scope.get("route")
    if route is not None and getattr(route, "path", None):
        return route.path
    return request.url.path


async def _metrics_middleware(request: Request, call_next: Any) -> Any:
    """Measure every request and feed the collector. Runs before the
    route handler, so it captures the full request lifecycle including
    middleware (CORS, etc.)."""
    start = time.perf_counter()
    status_code = 500  # default if the handler raises
    try:
        response = await call_next(request)
        status_code = response.status_code
        return response
    finally:
        duration = time.perf_counter() - start
        get_collector().observe(
            route=_route_template(request),
            method=request.method,
            status=status_code,
            duration_s=duration,
        )


def install_metrics(app: FastAPI) -> None:
    """Install the metrics middleware + the /metrics endpoint on `app`.

    Idempotent: safe to call once at startup.
    """

    @app.middleware("http")
    async def _middleware(request: Request, call_next: Any) -> Any:
        return await _metrics_middleware(request, call_next)

    @app.get("/metrics", include_in_schema=False)
    async def metrics_endpoint() -> PlainTextResponse:
        return PlainTextResponse(
            content=get_collector().render(),
            media_type="text/plain; version=0.0.4; charset=utf-8",
        )


__all__ = ["MetricsCollector", "get_collector", "install_metrics", "reset_collector"]

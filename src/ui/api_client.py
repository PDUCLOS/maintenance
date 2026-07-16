"""HTTP client for the FastAPI backend, used by the Streamlit UI.

Two helpers: api_get (for /health, /index/stats) and api_post
(for /query, /eval, /ingest). They return the parsed JSON body
or None on error. We use a small timeout (5s for GET, 60s for
POST) to avoid blocking the UI forever on a hung backend.

Streamlit-agnostic on purpose: callers (sidebar / tab renderers)
decide how to surface failures — a sidebar banner, a st.error in
the chat, or a quiet fallback. Keeping the client pure makes it
trivial to unit-test (httpx is patched at the transport layer).
"""

from __future__ import annotations

import httpx

from src.config import settings
from src.utils.logger import logger

# Browser-facing base URL. The API binds on 0.0.0.0 (so the host
# accepts external connections) but the browser always connects
# to localhost — binding 0.0.0.0 in the client base would resolve
# to the wrong host on some setups.
_BASE_URL = f"http://localhost:{settings.api_port}"


def api_get(path: str, timeout: float = 5.0) -> dict | None:
    """GET from the API. Returns the parsed JSON body, or None on error.

    Errors are logged but not raised — the UI should gracefully show
    a 'backend unreachable' state instead of crashing.
    """
    try:
        r = httpx.get(f"{_BASE_URL}{path}", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("GET {} failed: {}", path, e)
        return None


def api_post(path: str, payload: dict, timeout: float = 60.0) -> dict | None:
    """POST to the API. Returns the parsed JSON body, or None on error."""
    try:
        r = httpx.post(f"{_BASE_URL}{path}", json=payload, timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        logger.warning("POST {} failed: {}", path, e)
        return None


__all__ = ["api_get", "api_post"]

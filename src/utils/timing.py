"""Timing decorator — measures and logs wall-clock duration of a function call."""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from typing import Any, TypeVar

from src.utils.logger import logger

F = TypeVar("F", bound=Callable[..., Any])


def timed(func: F) -> F:
    """Decorator that logs how long `func` took to run."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            result = func(*args, **kwargs)
        except Exception:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.exception(f"{func.__qualname__} failed after {elapsed_ms:.1f} ms")
            raise
        else:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info(f"{func.__qualname__} completed in {elapsed_ms:.1f} ms")
            return result

    return wrapper  # type: ignore[return-value]


__all__ = ["timed"]

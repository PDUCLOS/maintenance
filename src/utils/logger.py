"""Centralized loguru configuration.

Structured JSON logs in production / CI, pretty console logs in dev.
Import `logger` everywhere — never instantiate loggers ad hoc.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from loguru import logger

from src.config import settings


def _json_sink(message) -> None:
    """loguru sink that emits one JSON object per record."""
    record = message.record
    payload = {
        "ts": record["time"].isoformat(),
        "level": record["level"].name,
        "msg": record["message"],
        "module": record["name"],
        "function": record["function"],
        "line": record["line"],
    }
    # Merge extra={"...": ...} fields if any
    if record["extra"]:
        payload.update(record["extra"])
    sys.stdout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    sys.stdout.flush()


def setup_logging() -> None:
    """Configure loguru sinks based on settings.log_format."""
    logger.remove()  # drop the default stderr sink

    if settings.log_format == "json":
        logger.add(
            _json_sink,
            level=settings.log_level,
            serialize=False,  # we serialize ourselves
        )
    else:
        logger.add(
            sys.stderr,
            level=settings.log_level,
            format=(
                "<green>{time:HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
                "<level>{message}</level>"
            ),
            colorize=True,
        )

    # Optional: also write to file under logs/
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    logger.add(
        log_dir / "app.log",
        level=settings.log_level,
        rotation="10 MB",
        retention="14 days",
        serialize=(settings.log_format == "json"),
    )


# Configure on import so `from src.utils.logger import logger` just works.
setup_logging()


__all__ = ["logger", "setup_logging"]

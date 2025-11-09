"""Centralized logging utilities for NetSec Tutor."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

LOG_DIR = "logs"
LOGGER_NAME = "netsec_tutor"
_LOGGER: Optional[logging.Logger] = None
_CURRENT_LOG_PATH: Optional[str] = None


def _determine_log_path() -> str:
    """Return the absolute path for today's log file."""
    filename = f"{datetime.now():%Y-%m-%d}.log"
    return os.path.join(LOG_DIR, filename)


def _configure_logger(logger: logging.Logger) -> None:
    """Ensure the logger writes only to the dated file handler."""
    global _CURRENT_LOG_PATH
    os.makedirs(LOG_DIR, exist_ok=True)
    target_path = _determine_log_path()
    if _CURRENT_LOG_PATH == target_path and logger.handlers:
        return

    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        handler.close()

    handler = logging.FileHandler(target_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    _CURRENT_LOG_PATH = target_path


def get_app_logger() -> logging.Logger:
    """Return the shared application logger."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = logging.getLogger(LOGGER_NAME)
    _configure_logger(_LOGGER)
    return _LOGGER


def summarize_text(text: Optional[str], max_len: int = 160) -> str:
    """Return a single-line summary suitable for logging."""
    if not text:
        return ""
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= max_len:
        return cleaned
    return f"{cleaned[:max_len]}…"


__all__ = ["get_app_logger", "summarize_text"]

"""
RevertIQ — Logging Configuration
=================================

Provides a pre-configured logger with both console and file output.

Usage
-----
>>> from revertiq.utils.logger import get_logger
>>> logger = get_logger(__name__)
>>> logger.info("Downloading NIFTY 50 data...")
"""

import logging
import os
import sys
from datetime import datetime
from typing import Optional


_LOGGERS: dict = {}  # Cache to avoid duplicate handlers


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Get or create a logger with console and (optional) file handlers.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    level : int
        Logging level (default: INFO).
    log_dir : str, optional
        Directory for log files.  If *None*, file logging is skipped
        (Colab-friendly default).

    Returns
    -------
    logging.Logger
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # ── Console handler ──
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(level)
    console_fmt = logging.Formatter(
        "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )
    console.setFormatter(console_fmt)
    logger.addHandler(console)

    # ── File handler (optional) ──
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        today = datetime.now().strftime("%Y-%m-%d")
        filepath = os.path.join(log_dir, f"revertiq_{today}.log")
        file_handler = logging.FileHandler(filepath, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_fmt = logging.Formatter(
            "%(asctime)s │ %(levelname)-8s │ %(name)s │ %(funcName)s:%(lineno)d │ %(message)s",
        )
        file_handler.setFormatter(file_fmt)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger

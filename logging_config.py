"""Dual logging: user-facing console output and system logs under logs/."""

from __future__ import annotations

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_DIR = Path(__file__).resolve().parent / "logs"
SYSTEM_LOG_FILE = LOG_DIR / "system.log"

_USER_LOGGER_NAME = "laptop_spec_agent.user"
_SYSTEM_LOGGER_NAME = "laptop_spec_agent.system"
_CONFIGURED = False


def setup_logging() -> tuple[logging.Logger, logging.Logger]:
    """Configure console and file loggers for the agent (idempotent).

    Creates ``logs/system.log`` with rotation. User logger prints plain messages
    to stdout; system logger records debug detail with timestamps.

    Returns:
        Tuple of ``(user_logger, system_logger)``.

    Side effects:
        Attaches handlers on first call; later calls return existing loggers.
    """
    global _CONFIGURED

    user_logger = logging.getLogger(_USER_LOGGER_NAME)
    system_logger = logging.getLogger(_SYSTEM_LOGGER_NAME)

    if _CONFIGURED:
        return user_logger, system_logger

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()
    system_logger.setLevel(getattr(logging, log_level, logging.DEBUG))
    user_logger.setLevel(logging.INFO)
    user_logger.propagate = False
    system_logger.propagate = False

    user_handler = logging.StreamHandler()
    user_handler.setLevel(logging.INFO)
    user_handler.setFormatter(logging.Formatter("%(message)s"))

    system_handler = RotatingFileHandler(
        SYSTEM_LOG_FILE,
        maxBytes=int(os.getenv("LOG_MAX_BYTES", str(5 * 1024 * 1024))),
        backupCount=int(os.getenv("LOG_BACKUP_COUNT", "5")),
        encoding="utf-8",
    )
    system_handler.setLevel(logging.DEBUG)
    system_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s | %(levelname)-8s | %(name)s | "
            "%(module)s:%(lineno)d | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ),
    )

    user_logger.addHandler(user_handler)
    system_logger.addHandler(system_handler)

    _CONFIGURED = True
    system_logger.debug("Logging initialized; system log at %s", SYSTEM_LOG_FILE)
    return user_logger, system_logger


def get_user_logger() -> logging.Logger:
    """Return the logger used for concise progress messages on stdout.

    Returns:
        Logger named ``laptop_spec_agent.user`` (handlers configured via
        ``setup_logging``).
    """
    setup_logging()
    return logging.getLogger(_USER_LOGGER_NAME)


def get_system_logger() -> logging.Logger:
    """Return the logger used for detailed traces in ``logs/system.log``.

    Returns:
        Logger named ``laptop_spec_agent.system`` (handlers configured via
        ``setup_logging``).
    """
    setup_logging()
    return logging.getLogger(_SYSTEM_LOGGER_NAME)

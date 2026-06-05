"""Logging bootstrap (loguru) and a small key=value event helper for observability.

Logs are emitted to stderr so they never mix with the styled CLI output on stdout.
Each module binds a short ``name`` (e.g. ``graph``, ``tools``, ``polymarket``) that loguru
renders in the log line. Loguru's default level severities match the stdlib ``logging``
constants, so call sites can keep passing ``logging.INFO`` / ``logging.DEBUG``.
"""

from __future__ import annotations

import logging
import sys

from loguru import logger

from config import Settings, get_settings

# Map stdlib logging int levels to loguru level names so call sites can keep using
# logging.INFO / logging.DEBUG while loguru renders proper named, colored levels.
_LEVEL_NAMES = {
    logging.DEBUG: "DEBUG",
    logging.INFO: "INFO",
    logging.WARNING: "WARNING",
    logging.ERROR: "ERROR",
    logging.CRITICAL: "CRITICAL",
}

_LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> "
    "<level>{level: <8}</level> "
    "<cyan>football_agent.{extra[name]}</cyan> "
    "<level>{message}</level>"
)

_configured = False


def configure_logging(settings: Settings | None = None, *, verbose: bool = False) -> None:
    """Configure loguru to write grep-friendly lines to stderr once per process."""
    global _configured
    settings = settings or get_settings()
    level = "DEBUG" if verbose else settings.log_level

    logger.remove()
    logger.configure(extra={"name": "main"})
    logger.add(sys.stderr, level=level, format=_LOG_FORMAT, colorize=True)
    _configured = True


def get_logger(name: str):
    """Return a loguru logger bound with a short module ``name``."""
    return logger.bind(name=name)


def _format_value(value: object) -> str:
    text = str(value)
    if len(text) > 120:
        text = text[:117] + "..."
    if " " in text:
        return f'"{text}"'
    return text


def log_event(bound_logger, level: int, event: str, **fields: object) -> None:
    """Emit a single grep-friendly ``event=... key=value`` line.

    ``level`` accepts a stdlib ``logging`` constant (e.g. ``logging.INFO``); loguru's default
    severities share the same numeric values.
    """
    parts = [f"event={event}"]
    parts.extend(
        f"{key}={_format_value(value)}"
        for key, value in fields.items()
        if value is not None
    )
    level_name = _LEVEL_NAMES.get(level, "INFO")
    # No positional/keyword args are passed, so loguru does not treat the message as a
    # format string -- curly braces in values (e.g. tool args) are safe.
    bound_logger.log(level_name, " ".join(parts))

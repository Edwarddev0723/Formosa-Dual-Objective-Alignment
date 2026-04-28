"""formosa_dual.utils.logging — project-wide logger factory.

All library modules MUST obtain their logger through ``get_logger()``.
No ``print()`` calls are allowed inside library code.
"""
import logging
import os
import sys

_HANDLER_ATTACHED: set[str] = set()


def get_logger(name: str = "formosa_dual") -> logging.Logger:
    """Return a logger configured for the project.

    The logger is idempotent: calling this function multiple times with the
    same ``name`` will not attach duplicate handlers.

    Configuration:
        - Level: ``FORMOSA_LOG_LEVEL`` env-var (default ``INFO``).
        - Format: ``[{asctime}] {levelname} {name}: {message}``
        - Stream: ``stderr``

    Args:
        name: Logger name, typically ``__name__`` of the calling module.

    Returns:
        Configured :class:`logging.Logger`.
    """
    logger = logging.getLogger(name)

    level_str = os.environ.get("FORMOSA_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_str, logging.INFO)
    logger.setLevel(level)

    if name not in _HANDLER_ATTACHED:
        handler = logging.StreamHandler(sys.stderr)
        handler.setLevel(level)
        formatter = logging.Formatter(
            fmt="[{asctime}] {levelname} {name}: {message}",
            style="{",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        _HANDLER_ATTACHED.add(name)

    return logger

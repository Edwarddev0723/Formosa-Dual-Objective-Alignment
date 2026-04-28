"""formosa_dual.utils.timing — wall-clock timer context manager."""
import time

from formosa_dual.utils.logging import get_logger

logger = get_logger(__name__)


class Timer:
    """Context manager that logs elapsed wall-clock time on exit.

    Example::

        with Timer("model forward"):
            output = model(batch)
    """

    def __init__(self, label: str = "") -> None:
        self.label = label
        self._start: float = 0.0
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_) -> None:
        self.elapsed = time.perf_counter() - self._start
        logger.debug("Timer[%s] elapsed %.3fs", self.label, self.elapsed)

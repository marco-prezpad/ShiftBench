"""
logging.py

Logging and timing utilities.

Author: Marco Pérez Padilla
Date:   09-08-2026
"""

import logging
import time
from contextlib import contextmanager
from typing import Generator


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the root logger for ShiftBench."""
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("shiftbench")


@contextmanager
def timer(name: str = "block") -> Generator[None, None, None]:
    """Context manager to log elapsed time."""
    logger = logging.getLogger("shiftbench")
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    logger.info(f"[TIMER] {name}: {elapsed:.3f}s")
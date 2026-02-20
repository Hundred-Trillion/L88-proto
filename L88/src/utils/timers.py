# L88/src/utils/timers.py
"""Simple timing utilities."""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator


@contextmanager
def timer() -> Generator[dict, None, None]:
    """Measure elapsed time with context manager."""
    payload = {"start": time.perf_counter(), "elapsed": 0.0}
    try:
        yield payload
    finally:
        payload["elapsed"] = time.perf_counter() - payload["start"]

# L88/src/optimization/scheduler.py
"""Lightweight scheduling helper for bounded loops."""

from __future__ import annotations


def can_retry(current_retry: int, max_retries: int = 2) -> bool:
    """Whether another iteration is allowed."""
    return current_retry < max_retries

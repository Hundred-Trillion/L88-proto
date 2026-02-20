# L88/src/optimization/fingerprint.py
"""Stable hashing utilities."""

from __future__ import annotations

import hashlib


def fingerprint_text(text: str) -> str:
    """Create deterministic hash key for text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

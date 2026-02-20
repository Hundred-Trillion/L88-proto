# L88/src/optimization/router.py
"""Routing logic for direct answer vs retrieval pipeline."""

from __future__ import annotations


def needs_retrieval(query: str) -> bool:
    """Rule-based decision for retrieval path."""
    keywords = ("who", "what", "where", "when", "why", "how", "source", "cite")
    lowered = query.lower()
    return any(k in lowered for k in keywords) or len(query.split()) > 6

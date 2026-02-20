# L88/src/storage/chunk_cache.py
"""In-memory chunk cache."""

from __future__ import annotations

from typing import Dict, List


class ChunkCache:
    """Caches chunked document outputs by document id."""

    def __init__(self) -> None:
        self._cache: Dict[str, List[str]] = {}

    def get(self, doc_id: str) -> List[str] | None:
        return self._cache.get(doc_id)

    def set(self, doc_id: str, chunks: List[str]) -> None:
        self._cache[doc_id] = chunks

    def clear(self) -> None:
        self._cache.clear()

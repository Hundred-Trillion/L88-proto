# L88/src/optimization/cache_manager.py
"""Result cache for repeated prompts and retrievals."""

from __future__ import annotations

from typing import Any, Dict


class CacheManager:
    """Simple key-value cache."""

    def __init__(self) -> None:
        self._cache: Dict[str, Any] = {}

    def get(self, key: str) -> Any:
        return self._cache.get(key)

    def set(self, key: str, value: Any) -> None:
        self._cache[key] = value

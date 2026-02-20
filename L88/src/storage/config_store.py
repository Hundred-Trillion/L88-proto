# L88/src/storage/config_store.py
"""Lightweight runtime config store."""

from __future__ import annotations

from typing import Any, Dict


class ConfigStore:
    """In-memory mutable config for runtime tuning."""

    def __init__(self) -> None:
        self._values: Dict[str, Any] = {}

    def set(self, key: str, value: Any) -> None:
        self._values[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._values.get(key, default)

    def all(self) -> Dict[str, Any]:
        return dict(self._values)

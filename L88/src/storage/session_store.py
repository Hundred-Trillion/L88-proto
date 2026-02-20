# L88/src/storage/session_store.py
"""Simple in-memory session state store."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class SessionData:
    """Session state for user interactions."""

    queries: List[str] = field(default_factory=list)


class SessionStore:
    """Store state by session id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionData] = {}

    def get_or_create(self, session_id: str) -> SessionData:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionData()
        return self._sessions[session_id]

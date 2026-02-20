# L88/src/api/deps.py
"""Dependency container for API routes."""

from __future__ import annotations

from functools import lru_cache

from src.agent.graph import StrongAgentGraph
from src.retrieval.chunker import TextChunker
from src.retrieval.retrieval_engine import RetrievalEngine
from src.storage.document_store import DocumentStore


@lru_cache(maxsize=1)
def get_retrieval_engine() -> RetrievalEngine:
    return RetrievalEngine()


@lru_cache(maxsize=1)
def get_agent_graph() -> StrongAgentGraph:
    return StrongAgentGraph(get_retrieval_engine())


@lru_cache(maxsize=1)
def get_chunker() -> TextChunker:
    return TextChunker()


@lru_cache(maxsize=1)
def get_document_store() -> DocumentStore:
    return DocumentStore()

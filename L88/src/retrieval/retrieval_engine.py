# L88/src/retrieval/retrieval_engine.py
"""Unified orchestration for dense+sparse+rerank retrieval."""

from __future__ import annotations

from typing import List, Tuple

from src.retrieval.bm25 import BM25Retriever
from src.retrieval.embedder import Embedder
from src.retrieval.faiss_index import FaissIndex
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import Reranker
from src.utils.config import get_settings


class RetrievalEngine:
    """Main retrieval facade used by the agent and API."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.embedder = Embedder()
        self.faiss = FaissIndex()
        self.bm25 = BM25Retriever()
        self.reranker = Reranker()

        self.faiss.load()
        self.bm25.load()

    def build_indexes(self, chunks: List[str]) -> None:
        embeddings = self.embedder.encode(chunks)
        self.faiss.build(embeddings, chunks)
        self.bm25.build(chunks)

    def retrieve(self, query: str) -> List[Tuple[str, float]]:
        """Run hybrid retrieval and return reranked evidence tuples."""
        q_emb = self.embedder.encode([query])
        dense = self.faiss.search(q_emb, self.settings.top_k_dense)
        sparse = self.bm25.search(query, self.settings.top_k_sparse)
        fused = reciprocal_rank_fusion([dense, sparse])[: self.settings.top_k_fused]
        reranked = self.reranker.rerank(query, fused, self.settings.top_k_rerank)
        return reranked

    def status(self) -> dict:
        return {
            "faiss_size": self.faiss.size(),
            "bm25_ready": self.bm25.model is not None,
        }

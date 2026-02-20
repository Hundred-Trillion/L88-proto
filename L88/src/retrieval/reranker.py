# L88/src/retrieval/reranker.py
"""Cross-encoder reranker for fused retrieval results."""

from __future__ import annotations

from typing import List, Tuple

from sentence_transformers import CrossEncoder

from src.utils.config import get_settings


class Reranker:
    """Rerank query-document pairs using bge-reranker-large."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.reranker_model_name
        self.model = CrossEncoder(self.model_name, device="cuda")

    def rerank(self, query: str, docs: List[Tuple[str, float]], top_k: int) -> List[Tuple[str, float]]:
        if not docs:
            return []
        pairs = [[query, text] for text, _ in docs]
        scores = self.model.predict(pairs)
        rescored = [(docs[i][0], float(scores[i])) for i in range(len(docs))]
        rescored.sort(key=lambda x: x[1], reverse=True)
        return rescored[:top_k]

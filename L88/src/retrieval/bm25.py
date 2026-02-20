# L88/src/retrieval/bm25.py
"""Sparse retrieval with BM25."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Tuple

from rank_bm25 import BM25Okapi

from src.utils.config import get_settings
from src.utils.text_utils import simple_tokenize


class BM25Retriever:
    """Build and query BM25 over chunk texts."""

    def __init__(self, path: Path | None = None) -> None:
        settings = get_settings()
        self.path = path or settings.bm25_path
        self.texts: List[str] = []
        self.model: BM25Okapi | None = None

    def build(self, texts: List[str]) -> None:
        self.texts = texts
        tokenized = [simple_tokenize(t) for t in texts]
        self.model = BM25Okapi(tokenized)
        self.save()

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        if self.model is None:
            return []
        scores = self.model.get_scores(simple_tokenize(query))
        pairs = list(enumerate(scores))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return [(self.texts[idx], float(score)) for idx, score in pairs[:top_k]]

    def save(self) -> None:
        with self.path.open("wb") as f:
            pickle.dump({"texts": self.texts, "model": self.model}, f)

    def load(self) -> bool:
        if not self.path.exists():
            return False
        with self.path.open("rb") as f:
            data = pickle.load(f)
        self.texts = data["texts"]
        self.model = data["model"]
        return True

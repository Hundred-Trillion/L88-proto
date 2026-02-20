# L88/src/retrieval/embedder.py
"""GPU embedding encoder backed by SentenceTransformer."""

from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from src.utils.config import get_settings


class Embedder:
    """Wrapper around BGE embedding model."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.embedding_model_name
        self.model = SentenceTransformer(self.model_name, device="cuda")

    def encode(self, texts: List[str]) -> np.ndarray:
        """Encode texts into float32 embeddings."""
        if not texts:
            return np.zeros((0, 1024), dtype=np.float32)
        emb = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(emb, dtype=np.float32)

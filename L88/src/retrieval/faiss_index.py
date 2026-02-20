# L88/src/retrieval/faiss_index.py
"""FAISS GPU index management."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

from src.utils.config import get_settings
from src.utils.logger import ensure_directory


class FaissIndex:
    """Manages dense vector search with FAISS GPU (fallback CPU)."""

    def __init__(self, index_path: Path | None = None, meta_path: Path | None = None) -> None:
        settings = get_settings()
        self.index_path = index_path or settings.vector_index_path
        self.meta_path = meta_path or settings.vector_meta_path
        ensure_directory(self.index_path.parent)
        self.index: faiss.Index | None = None
        self.texts: List[str] = []

    def build(self, embeddings: np.ndarray, texts: List[str]) -> None:
        """Create GPU FlatL2 index and persist to disk."""
        if len(texts) != embeddings.shape[0]:
            raise ValueError("texts and embeddings length mismatch")
        dim = embeddings.shape[1]
        cpu_index = faiss.IndexFlatL2(dim)
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception:
            self.index = cpu_index
        self.index.add(embeddings)
        self.texts = texts
        self.save()

    def save(self) -> None:
        """Persist index to disk."""
        if self.index is None:
            return
        cpu_index = faiss.index_gpu_to_cpu(self.index) if hasattr(faiss, "index_gpu_to_cpu") else self.index
        faiss.write_index(cpu_index, str(self.index_path))
        self.meta_path.write_text(json.dumps({"texts": self.texts}, ensure_ascii=False), encoding="utf-8")

    def load(self) -> bool:
        """Load index and metadata from disk if available."""
        if not self.index_path.exists() or not self.meta_path.exists():
            return False
        cpu_index = faiss.read_index(str(self.index_path))
        try:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        except Exception:
            self.index = cpu_index
        data = json.loads(self.meta_path.read_text(encoding="utf-8"))
        self.texts = data["texts"]
        return True

    def search(self, query_embedding: np.ndarray, top_k: int) -> List[Tuple[str, float]]:
        """Search nearest chunks."""
        if self.index is None:
            return []
        distances, indices = self.index.search(query_embedding, top_k)
        out: List[Tuple[str, float]] = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0 or idx >= len(self.texts):
                continue
            score = float(1.0 / (1.0 + dist))
            out.append((self.texts[idx], score))
        return out

    def size(self) -> int:
        return 0 if self.index is None else self.index.ntotal

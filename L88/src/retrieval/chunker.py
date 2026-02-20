# L88/src/retrieval/chunker.py
"""Text chunking for ingestion."""

from __future__ import annotations

from typing import List

from src.utils.config import get_settings
from src.utils.text_utils import normalize_text


class TextChunker:
    """Create overlapping chunks by character windows."""

    def __init__(self, chunk_size: int | None = None, chunk_overlap: int | None = None) -> None:
        settings = get_settings()
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

    def chunk(self, text: str) -> List[str]:
        """Split text into overlap-aware chunks."""
        text = normalize_text(text)
        if not text:
            return []

        chunks: List[str] = []
        start = 0
        while start < len(text):
            end = min(len(text), start + self.chunk_size)
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = max(0, end - self.chunk_overlap)
        return chunks

# L88/src/utils/text_utils.py
"""Utility helpers for text normalization and tokenization."""

from __future__ import annotations

import re
from typing import List


_WHITESPACE = re.compile(r"\s+")
_WORD = re.compile(r"\b\w+\b")


def normalize_text(text: str) -> str:
    """Normalize whitespace and strip control chars."""
    return _WHITESPACE.sub(" ", text.replace("\x00", " ")).strip()


def simple_tokenize(text: str) -> List[str]:
    """Lightweight tokenizer for BM25 indexing."""
    return [tok.lower() for tok in _WORD.findall(normalize_text(text))]

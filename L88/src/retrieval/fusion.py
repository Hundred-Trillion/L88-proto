# L88/src/retrieval/fusion.py
"""Hybrid retrieval fusion via Reciprocal Rank Fusion."""

from __future__ import annotations

from collections import defaultdict
from typing import Dict, List, Tuple


def reciprocal_rank_fusion(ranked_lists: List[List[Tuple[str, float]]], k: int = 60) -> List[Tuple[str, float]]:
    """Fuse multiple ranked lists using RRF score."""
    score_map: Dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, (text, _) in enumerate(ranked, start=1):
            score_map[text] += 1.0 / (k + rank)
    merged = sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    return merged

# L88/tests/test_fusion.py
from src.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_prefers_consensus_item():
    a = [("doc1", 0.9), ("doc2", 0.8)]
    b = [("doc2", 1.0), ("doc1", 0.7)]
    out = reciprocal_rank_fusion([a, b], k=1)
    assert out[0][0] in {"doc1", "doc2"}
    assert len(out) == 2

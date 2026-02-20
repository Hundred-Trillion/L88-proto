# L88/src/api/routes/index.py
"""Index rebuild route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_chunker, get_document_store, get_retrieval_engine

router = APIRouter(prefix="/reindex", tags=["index"])


@router.post("")
def rebuild_index(
    chunker=Depends(get_chunker),
    store=Depends(get_document_store),
    retrieval_engine=Depends(get_retrieval_engine),
) -> dict:
    """Rebuild FAISS and BM25 indexes from stored documents."""
    docs = store.list_documents()
    chunks = []
    for doc in docs:
        chunks.extend(chunker.chunk(doc["text"]))
    retrieval_engine.build_indexes(chunks)
    return {"indexed_documents": len(docs), "indexed_chunks": len(chunks)}

# L88/src/api/routes/documents.py
"""Document upload and ingestion endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_chunker, get_document_store, get_retrieval_engine
from src.api.schemas.common import UploadRequest

router = APIRouter(prefix="/upload", tags=["documents"])


@router.post("")
def upload_documents(
    payload: UploadRequest,
    chunker=Depends(get_chunker),
    store=Depends(get_document_store),
    retrieval_engine=Depends(get_retrieval_engine),
) -> dict:
    """Store incoming docs and update retrieval indexes."""
    docs = [{"id": f"doc_{i}", "text": text} for i, text in enumerate(payload.documents)]
    store.add_documents(docs)

    all_docs = store.list_documents()
    chunks = []
    for doc in all_docs:
        chunks.extend(chunker.chunk(doc["text"]))

    retrieval_engine.build_indexes(chunks)
    return {"documents": len(payload.documents), "total_chunks": len(chunks)}

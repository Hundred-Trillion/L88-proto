# L88/src/api/routes/system.py
"""System health route."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_retrieval_engine
from src.api.schemas.common import HealthResponse

router = APIRouter(prefix="/health", tags=["system"])


@router.get("", response_model=HealthResponse)
def health(retrieval_engine=Depends(get_retrieval_engine)) -> HealthResponse:
    """Return system readiness summary."""
    details = retrieval_engine.status()
    return HealthResponse(status="ok", details=details)

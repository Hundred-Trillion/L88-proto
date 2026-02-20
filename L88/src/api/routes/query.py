# L88/src/api/routes/query.py
"""Query endpoint routing."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from src.api.deps import get_agent_graph
from src.api.schemas.common import QueryRequest, QueryResponse

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def run_query(payload: QueryRequest, agent=Depends(get_agent_graph)) -> QueryResponse:
    """Run full plan-act-verify-synthesize agent flow."""
    result = agent.run(payload.query)
    answer = result.get("answer", {})
    return QueryResponse(
        answer=answer.get("answer", ""),
        citations=answer.get("citations", []),
        plan=result.get("plan", {}),
        verification=result.get("verification", {}),
    )

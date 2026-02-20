# L88/src/api/main.py
"""FastAPI entrypoint for L88 local agentic RAG."""

from __future__ import annotations

from fastapi import FastAPI

from src.api.middleware.request_id import RequestIDMiddleware
from src.api.routes.documents import router as documents_router
from src.api.routes.index import router as index_router
from src.api.routes.query import router as query_router
from src.api.routes.system import router as system_router
from src.utils.config import get_settings


def create_app() -> FastAPI:
    """Build and configure FastAPI app."""
    settings = get_settings()
    app = FastAPI(title=settings.project_name)
    app.add_middleware(RequestIDMiddleware)

    app.include_router(query_router)
    app.include_router(documents_router)
    app.include_router(index_router)
    app.include_router(system_router)

    return app


app = create_app()

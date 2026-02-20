# L88/src/api/middleware/request_id.py
"""Attach request ids for observability."""

from __future__ import annotations

import uuid

from starlette.middleware.base import BaseHTTPMiddleware


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Inject X-Request-ID header."""

    async def dispatch(self, request, call_next):
        request_id = str(uuid.uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

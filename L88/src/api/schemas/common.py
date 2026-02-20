# L88/src/api/schemas/common.py
"""Pydantic schemas for API."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(min_length=1)


class QueryResponse(BaseModel):
    answer: str
    citations: list[str]
    plan: dict
    verification: dict


class UploadRequest(BaseModel):
    documents: list[str]


class HealthResponse(BaseModel):
    status: str
    details: dict

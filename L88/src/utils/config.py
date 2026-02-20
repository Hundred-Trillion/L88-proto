# L88/src/utils/config.py
"""Runtime configuration using environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""

    project_name: str = "L88 Agentic RAG"
    data_dir: Path = Field(default=Path("L88/data"))
    models_dir: Path = Field(default=Path("L88/models"))
    config_dir: Path = Field(default=Path("L88/configs"))

    embedding_model_name: str = "BAAI/bge-large-en-v1.5"
    reranker_model_name: str = "BAAI/bge-reranker-large"

    llm_model: str = "Qwen/Qwen2.5-7B-Instruct"
    hyde_model: str = "microsoft/Phi-3-mini-4k-instruct"

    vector_index_path: Path = Field(default=Path("L88/data/faiss.index"))
    vector_meta_path: Path = Field(default=Path("L88/data/faiss_meta.json"))
    bm25_path: Path = Field(default=Path("L88/data/bm25.pkl"))
    documents_path: Path = Field(default=Path("L88/data/documents.jsonl"))

    chunk_size: int = 700
    chunk_overlap: int = 120

    top_k_dense: int = 10
    top_k_sparse: int = 10
    top_k_fused: int = 10
    top_k_rerank: int = 5

    vllm_temperature: float = 0.2
    vllm_top_p: float = 0.9
    max_tokens: int = 512

    api_host: str = "0.0.0.0"
    api_port: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Get singleton settings."""
    return Settings()

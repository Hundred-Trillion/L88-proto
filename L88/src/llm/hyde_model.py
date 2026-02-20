# L88/src/llm/hyde_model.py
"""HyDE helper model for synthetic evidence generation."""

from __future__ import annotations

from src.llm.model_config import SamplingConfig
from src.llm.vllm_client import VLLMClient
from src.utils.config import get_settings


class HyDEModel:
    """Generate synthetic passage for retrieval expansion."""

    def __init__(self, model_name: str | None = None) -> None:
        settings = get_settings()
        self.client = VLLMClient(
            model_name=model_name or settings.hyde_model,
            sampling=SamplingConfig(temperature=0.2, top_p=0.9, max_tokens=256),
        )

    def generate_hypothesis(self, query: str) -> str:
        prompt = (
            "Write a concise factual passage that would answer the user query. "
            "Do not mention this is hypothetical.\n"
            f"Query: {query}\nPassage:"
        )
        return self.client.generate(prompt)

# L88/src/llm/vllm_client.py
"""vLLM local inference wrapper."""

from __future__ import annotations

from vllm import LLM, SamplingParams

from src.llm.model_config import SamplingConfig
from src.utils.config import get_settings


class VLLMClient:
    """Text generation client backed by vLLM."""

    def __init__(self, model_name: str | None = None, sampling: SamplingConfig | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.llm_model
        self.sampling = sampling or SamplingConfig(
            temperature=settings.vllm_temperature,
            top_p=settings.vllm_top_p,
            max_tokens=settings.max_tokens,
        )
        self.llm = LLM(model=self.model_name, dtype="float16")

    def generate(self, prompt: str) -> str:
        params = SamplingParams(
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            max_tokens=self.sampling.max_tokens,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

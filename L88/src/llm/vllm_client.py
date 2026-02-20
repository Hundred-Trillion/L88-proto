# L88/src/llm/vllm_client.py
"""vLLM local inference wrapper with shared engine instances."""

from __future__ import annotations

from threading import Lock

from vllm import LLM, SamplingParams

from src.llm.model_config import SamplingConfig
from src.utils.config import get_settings


class VLLMClient:
    """Text generation client backed by vLLM.

    Multiple ``VLLMClient`` objects share one loaded ``LLM`` per model name
    to avoid repeatedly allocating full model weights in GPU memory.
    """

    _engine_cache: dict[str, LLM] = {}
    _cache_lock = Lock()

    def __init__(self, model_name: str | None = None, sampling: SamplingConfig | None = None) -> None:
        settings = get_settings()
        self.model_name = model_name or settings.llm_model
        self.sampling = sampling or SamplingConfig(
            temperature=settings.vllm_temperature,
            top_p=settings.vllm_top_p,
            max_tokens=settings.max_tokens,
        )
        self.llm = self._get_or_create_engine(self.model_name)

    @classmethod
    def _get_or_create_engine(cls, model_name: str) -> LLM:
        """Get shared LLM engine instance for a model, creating once."""
        with cls._cache_lock:
            if model_name not in cls._engine_cache:
                cls._engine_cache[model_name] = LLM(model=model_name, dtype="float16")
            return cls._engine_cache[model_name]

    def generate(self, prompt: str) -> str:
        """Generate text with stable sampling settings."""
        params = SamplingParams(
            temperature=self.sampling.temperature,
            top_p=self.sampling.top_p,
            max_tokens=self.sampling.max_tokens,
        )
        outputs = self.llm.generate([prompt], params)
        return outputs[0].outputs[0].text.strip()

# L88/src/llm/model_config.py
"""Model sampling and engine configuration."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SamplingConfig:
    temperature: float = 0.2
    top_p: float = 0.9
    max_tokens: int = 512

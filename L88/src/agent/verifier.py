# L88/src/agent/verifier.py
"""VERIFY node implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.llm.vllm_client import VLLMClient
from src.optimization.scheduler import can_retry


class Verifier:
    """Checks evidence sufficiency and retry signal."""

    def __init__(self) -> None:
        self.llm = VLLMClient()
        self.prompt = (Path(__file__).resolve().parent / "prompts" / "evidence_prompt.txt").read_text(encoding="utf-8")

    def run(self, query: str, evidence: list, retry_count: int) -> Dict:
        short_evidence = "\n".join([f"- {e['text'][:300]}" for e in evidence[:5]])
        raw = self.llm.generate(
            f"{self.prompt}\nQuery: {query}\nEvidence:\n{short_evidence}\nReturn JSON only."
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"quality_score": 0.0, "is_sufficient": False, "missing_points": ["parse_failure"]}
        quality = float(data.get("quality_score", 0.0))
        sufficient = bool(data.get("is_sufficient", False)) or quality >= 0.7
        retry = (not sufficient) and can_retry(retry_count, max_retries=2)
        return {
            "quality_score": quality,
            "is_sufficient": sufficient,
            "retry": retry,
            "missing_points": data.get("missing_points", []),
        }

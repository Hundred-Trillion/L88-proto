# L88/src/agent/synthesizer.py
"""SYNTHESIZE node implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.llm.vllm_client import VLLMClient


class Synthesizer:
    """Builds final grounded answer from evidence."""

    def __init__(self) -> None:
        self.llm = VLLMClient()
        self.prompt = (Path(__file__).resolve().parent / "prompts" / "synthesis_prompt.txt").read_text(encoding="utf-8")

    def run(self, query: str, evidence: list) -> Dict:
        ev = "\n".join([f"[{i+1}] {e['text']}" for i, e in enumerate(evidence[:5])])
        raw = self.llm.generate(
            f"{self.prompt}\nQuery: {query}\nEvidence:\n{ev}\nReturn JSON only."
        )
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {
                "answer": "I could not synthesize a reliable answer from the evidence.",
                "citations": [e["text"][:120] for e in evidence[:2]],
            }
        return data

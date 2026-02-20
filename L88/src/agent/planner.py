# L88/src/agent/planner.py
"""PLAN node implementation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from src.llm.vllm_client import VLLMClient
from src.optimization.router import needs_retrieval


class Planner:
    """Creates plan actions for the agent graph."""

    def __init__(self) -> None:
        prompt_path = Path("L88/src/agent/prompts/planner_prompt.txt")
        self.system_prompt = prompt_path.read_text(encoding="utf-8")
        self.llm = VLLMClient()

    def run(self, query: str) -> Dict[str, str]:
        """Return planning decision JSON."""
        default_action = "retrieve" if needs_retrieval(query) else "direct_answer"
        prompt = (
            f"{self.system_prompt}\n"
            f"User query: {query}\n"
            f"Heuristic action: {default_action}\n"
            "Return valid JSON only."
        )
        raw = self.llm.generate(prompt)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = {"action": default_action, "reason": "fallback", "refined_query": query}
        data.setdefault("action", default_action)
        data.setdefault("refined_query", query)
        data.setdefault("reason", "model")
        return data

# L88/src/agent/executor.py
"""ACT node implementation."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

from src.llm.hyde_model import HyDEModel
from src.llm.vllm_client import VLLMClient
from src.retrieval.retrieval_engine import RetrievalEngine


class Executor:
    """Executes retrieval-related actions."""

    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.retrieval_engine = retrieval_engine
        self.hyde = HyDEModel()
        self.llm = VLLMClient()
        prompt_path = Path(__file__).resolve().parent / "prompts" / "refinement_prompt.txt"
        self.refinement_prompt = prompt_path.read_text(encoding="utf-8")

    def refine_query(self, query: str) -> str:
        prompt = f"{self.refinement_prompt}\nQuery: {query}\nJSON only:"
        out = self.llm.generate(prompt)
        if "refined_query" in out:
            try:
                import json

                return json.loads(out).get("refined_query", query)
            except Exception:
                return query
        return query

    def run(self, action: str, query: str) -> Dict:
        """Run ACT behavior based on planner action."""
        used_query = query
        evidence: List[Tuple[str, float]] = []

        if action == "expand_query":
            hypo = self.hyde.generate_hypothesis(query)
            used_query = f"{query}\n{hypo}"
            evidence = self.retrieval_engine.retrieve(used_query)
        elif action == "refine_query":
            used_query = self.refine_query(query)
            evidence = self.retrieval_engine.retrieve(used_query)
        elif action in {"retrieve", "final_answer"}:
            evidence = self.retrieval_engine.retrieve(query)

        return {
            "action": action,
            "query_used": used_query,
            "evidence": [{"text": t, "score": s} for t, s in evidence],
        }

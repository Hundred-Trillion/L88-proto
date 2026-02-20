# L88/src/agent/graph.py
"""LangGraph state machine for strong agent loop."""

from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, StateGraph

from src.agent.executor import Executor
from src.agent.planner import Planner
from src.agent.synthesizer import Synthesizer
from src.agent.verifier import Verifier
from src.retrieval.retrieval_engine import RetrievalEngine


class AgentState(TypedDict, total=False):
    query: str
    plan: dict
    act: dict
    verify: dict
    answer: dict
    retry_count: int


class StrongAgentGraph:
    """PLAN -> ACT -> VERIFY -> (ACT|SYNTHESIZE) pipeline."""

    def __init__(self, retrieval_engine: RetrievalEngine) -> None:
        self.planner = Planner()
        self.executor = Executor(retrieval_engine)
        self.verifier = Verifier()
        self.synthesizer = Synthesizer()
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgentState)

        def plan_node(state: AgentState) -> AgentState:
            plan = self.planner.run(state["query"])
            return {**state, "plan": plan}

        def act_node(state: AgentState) -> AgentState:
            action = state["plan"].get("action", "retrieve")
            query = state["plan"].get("refined_query", state["query"])
            act = self.executor.run(action, query)
            return {**state, "act": act}

        def verify_node(state: AgentState) -> AgentState:
            retry_count = state.get("retry_count", 0)
            verify = self.verifier.run(state["query"], state["act"].get("evidence", []), retry_count)
            state["retry_count"] = retry_count + (1 if verify.get("retry") else 0)
            return {**state, "verify": verify}

        def synthesize_node(state: AgentState) -> AgentState:
            answer = self.synthesizer.run(state["query"], state["act"].get("evidence", []))
            return {**state, "answer": answer}

        def route_after_verify(state: AgentState) -> str:
            if state["verify"].get("retry", False):
                return "ACT"
            return "SYNTHESIZE"

        graph.add_node("PLAN", plan_node)
        graph.add_node("ACT", act_node)
        graph.add_node("VERIFY", verify_node)
        graph.add_node("SYNTHESIZE", synthesize_node)

        graph.set_entry_point("PLAN")
        graph.add_edge("PLAN", "ACT")
        graph.add_edge("ACT", "VERIFY")
        graph.add_conditional_edges("VERIFY", route_after_verify, {"ACT": "ACT", "SYNTHESIZE": "SYNTHESIZE"})
        graph.add_edge("SYNTHESIZE", END)

        return graph.compile()

    def run(self, query: str) -> dict:
        state: AgentState = {"query": query, "retry_count": 0}
        out = self.graph.invoke(state)
        return {
            "plan": out.get("plan", {}),
            "evidence": out.get("act", {}).get("evidence", []),
            "verification": out.get("verify", {}),
            "answer": out.get("answer", {}),
        }

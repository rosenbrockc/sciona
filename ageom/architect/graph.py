"""LangGraph assembly and DecompositionAgent wrapper."""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from ageom.architect.handoff import CDGExport
from ageom.architect.models import NodeStatus
from ageom.architect.nodes import (
    advance_node,
    critique_decomposition,
    decompose_node,
    prepare_retry,
    route_after_advance,
    route_after_critic,
    select_strategy,
)
from ageom.architect.state import DecompositionDeps, DecompositionState
from ageom.hunter.llm import LLMClient


def build_graph() -> StateGraph:
    """Construct the decomposition StateGraph."""
    graph = StateGraph(DecompositionState)

    graph.add_node("select_strategy", select_strategy)
    graph.add_node("decompose_node", decompose_node)
    graph.add_node("critique", critique_decomposition)
    graph.add_node("advance_node", advance_node)
    graph.add_node("prepare_retry", prepare_retry)

    graph.set_entry_point("select_strategy")
    graph.add_edge("select_strategy", "decompose_node")
    graph.add_edge("decompose_node", "critique")
    graph.add_conditional_edges(
        "critique",
        route_after_critic,
        {"retry_decompose": "prepare_retry", "next_node": "advance_node"},
    )
    graph.add_edge("prepare_retry", "decompose_node")
    graph.add_conditional_edges(
        "advance_node",
        route_after_advance,
        {"decompose": "decompose_node", "end": END},
    )

    return graph


class DecompositionAgent:
    """High-level wrapper around the decomposition graph.

    Usage:
        agent = DecompositionAgent(catalog, skill_index, llm)
        cdg = await agent.decompose("Implement merge sort")
    """

    def __init__(
        self,
        catalog: Any,
        skill_index: Any,
        llm: LLMClient,
        max_depth: int = 8,
    ) -> None:
        self._deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
        )
        self._max_depth = max_depth
        self._graph = build_graph().compile()

    async def decompose(self, goal: str) -> CDGExport:
        """Decompose a high-level goal into a CDG.

        Returns a CDGExport with rejected nodes filtered out.
        """
        initial_state: dict[str, Any] = {
            "goal": goal,
            "max_depth": self._max_depth,
            "nodes": [],
            "edges": [],
            "history": [],
            "pending_node_ids": [],
            "current_node_id": "",
            "paradigm": "",
            "skeleton_instantiated": False,
            "critique_passed": False,
            "critique_reason": "",
            "critique_retries": 0,
            "done": False,
            "error": "",
        }

        config = {"configurable": {"deps": self._deps}}
        final_state = await self._graph.ainvoke(initial_state, config=config)

        # Filter out rejected nodes
        active_nodes = [
            n for n in final_state["nodes"]
            if n.status != NodeStatus.REJECTED
        ]

        return CDGExport(
            nodes=active_nodes,
            edges=final_state["edges"],
            metadata={
                "goal": goal,
                "paradigm": final_state.get("paradigm", ""),
                "max_depth": self._max_depth,
                "total_nodes_processed": len(final_state["nodes"]),
                "active_nodes": len(active_nodes),
                "history_steps": len(final_state.get("history", [])),
            },
        )

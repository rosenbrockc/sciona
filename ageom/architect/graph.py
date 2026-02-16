"""LangGraph assembly and DecompositionAgent wrapper."""

from __future__ import annotations

import uuid
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
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
        checkpointer: BaseCheckpointSaver | None = None,
    ) -> None:
        self._deps = DecompositionDeps(
            catalog=catalog,
            skill_index=skill_index,
            llm=llm,
        )
        self._max_depth = max_depth
        self._graph = build_graph().compile(checkpointer=checkpointer)

    async def decompose(
        self,
        goal: str,
        *,
        thread_id: str | None = None,
    ) -> CDGExport:
        """Decompose a high-level goal into a CDG.

        Args:
            goal: High-level goal to decompose.
            thread_id: Optional checkpoint thread identifier.
                Auto-generated as a 32-char hex string when None.

        Returns a CDGExport with rejected nodes filtered out.
        """
        if thread_id is None:
            thread_id = uuid.uuid4().hex

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

        config: dict[str, Any] = {
            "configurable": {
                "deps": self._deps,
                "thread_id": thread_id,
            }
        }
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
                "thread_id": thread_id,
            },
        )

    async def get_state(self, thread_id: str) -> dict:
        """Retrieve the latest checkpoint state for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        snapshot = await self._graph.aget_state(config)
        return {
            "values": snapshot.values,
            "checkpoint_id": snapshot.config["configurable"].get("checkpoint_id"),
        }

    async def get_state_history(self, thread_id: str) -> list[dict]:
        """Return all checkpoints for a thread, newest first."""
        config = {"configurable": {"thread_id": thread_id}}
        history: list[dict] = []
        async for snapshot in self._graph.aget_state_history(config):
            history.append({
                "values": snapshot.values,
                "checkpoint_id": snapshot.config["configurable"].get(
                    "checkpoint_id"
                ),
            })
        return history

    async def fork(
        self,
        source_thread_id: str,
        checkpoint_id: str,
        new_thread_id: str | None = None,
    ) -> str:
        """Fork a new thread from a specific checkpoint of an existing thread.

        Returns the new thread_id.
        """
        if new_thread_id is None:
            new_thread_id = uuid.uuid4().hex

        # Read state at the source checkpoint
        source_config = {
            "configurable": {
                "thread_id": source_thread_id,
                "checkpoint_id": checkpoint_id,
            }
        }
        snapshot = await self._graph.aget_state(source_config)

        # Write that state into the new thread
        new_config = {"configurable": {"thread_id": new_thread_id}}
        await self._graph.aupdate_state(new_config, snapshot.values)

        return new_thread_id

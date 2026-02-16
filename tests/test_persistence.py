"""Tests for Phase 3: state persistence, time-travel, and handoff validation.

All tests use MemorySaver — no PostgreSQL required.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.checkpointer import create_checkpointer
from ageom.architect.graph import DecompositionAgent
from ageom.architect.handoff import (
    CDGExport,
    HandoffValidationError,
    to_pdg_nodes,
    validate_handoff,
)
from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from langgraph.checkpoint.memory import MemorySaver


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_catalog() -> PrimitiveCatalog:
    catalog = PrimitiveCatalog()
    catalog.add(AlgorithmicPrimitive(
        name="merge",
        source="clrs-30",
        category=ConceptType.SORTING,
        description="Merge two sorted lists into one sorted list",
        inputs=[
            IOSpec(name="left", type_desc="list[comparable]"),
            IOSpec(name="right", type_desc="list[comparable]"),
        ],
        outputs=[IOSpec(name="result", type_desc="list[comparable]")],
        type_signature="list[T] -> list[T] -> list[T]",
    ))
    catalog.add(AlgorithmicPrimitive(
        name="compare",
        source="clrs-30",
        category=ConceptType.SORTING,
        description="Compare two elements",
        inputs=[
            IOSpec(name="a", type_desc="comparable"),
            IOSpec(name="b", type_desc="comparable"),
        ],
        outputs=[IOSpec(name="order", type_desc="bool")],
        type_signature="T -> T -> bool",
    ))
    return catalog


def _make_skill_index():
    index = AsyncMock()
    index.search = lambda query, k=10: []
    return index


def _make_mock_llm():
    """Mock LLM that drives a simple decomposition to completion."""
    strategy_response = json.dumps({
        "paradigm": "divide_and_conquer",
        "rationale": "Classic D&C",
        "variant_hint": "merge_sort",
    })
    decompose_response = json.dumps({
        "sub_nodes": [
            {
                "name": "Split Input",
                "description": "Split the input list into two halves",
                "concept_type": "divide_and_conquer",
                "inputs": [{"name": "data", "type_desc": "list[comparable]"}],
                "outputs": [
                    {"name": "left", "type_desc": "list[comparable]"},
                    {"name": "right", "type_desc": "list[comparable]"},
                ],
                "type_signature": "",
                "is_atomic": False,
                "matched_primitive": None,
            },
            {
                "name": "merge",
                "description": "Merge two sorted lists into one sorted list",
                "concept_type": "sorting",
                "inputs": [
                    {"name": "left", "type_desc": "list[comparable]"},
                    {"name": "right", "type_desc": "list[comparable]"},
                ],
                "outputs": [{"name": "result", "type_desc": "list[comparable]"}],
                "type_signature": "list[T] -> list[T] -> list[T]",
                "is_atomic": True,
                "matched_primitive": "merge",
            },
        ],
        "edges": [
            {
                "source_name": "Split Input",
                "target_name": "merge",
                "output_name": "left",
                "input_name": "left",
                "data_type": "list[comparable]",
            },
        ],
    })
    critique_response = json.dumps({
        "approved": True,
        "reason": "Looks good",
        "io_issues": [],
        "flagged_nodes": [],
    })

    llm = AsyncMock()

    async def complete(system: str, user: str) -> str:
        s = system.lower()
        if "critic" in s or "evaluate" in s:
            return critique_response
        elif "sub-nodes" in s or "sub_nodes" in s:
            return decompose_response
        elif "best" in s and "paradigm" in s:
            return strategy_response
        return "{}"

    llm.complete = complete
    return llm


def _build_agent(checkpointer=None):
    return DecompositionAgent(
        catalog=_make_catalog(),
        skill_index=_make_skill_index(),
        llm=_make_mock_llm(),
        max_depth=8,
        checkpointer=checkpointer,
    )


# ---------------------------------------------------------------------------
# TestCheckpointerFactory
# ---------------------------------------------------------------------------

class TestCheckpointerFactory:
    @pytest.mark.asyncio
    async def test_none_yields_memory_saver(self):
        async with create_checkpointer(None) as cp:
            assert isinstance(cp, MemorySaver)

    @pytest.mark.asyncio
    async def test_empty_string_yields_memory_saver(self):
        async with create_checkpointer("") as cp:
            assert isinstance(cp, MemorySaver)


# ---------------------------------------------------------------------------
# TestCheckpointPersistence
# ---------------------------------------------------------------------------

class TestCheckpointPersistence:
    @pytest.mark.asyncio
    async def test_thread_id_in_metadata(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)
        cdg = await agent.decompose("Sort a list", thread_id="test-thread-42")
        assert cdg.metadata["thread_id"] == "test-thread-42"

    @pytest.mark.asyncio
    async def test_get_state_returns_terminal(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)
        cdg = await agent.decompose("Sort a list", thread_id="terminal-check")
        state = await agent.get_state("terminal-check")
        assert state["values"]["done"] is True

    @pytest.mark.asyncio
    async def test_get_state_history_has_checkpoints(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)
        await agent.decompose("Sort a list", thread_id="history-check")
        history = await agent.get_state_history("history-check")
        # At minimum: initial state write, several node steps, terminal
        assert len(history) >= 3

    @pytest.mark.asyncio
    async def test_auto_thread_id_is_hex(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)
        cdg = await agent.decompose("Sort a list")
        tid = cdg.metadata["thread_id"]
        assert len(tid) == 32
        int(tid, 16)  # Raises if not valid hex


# ---------------------------------------------------------------------------
# TestTimeTravelAndFork
# ---------------------------------------------------------------------------

class TestTimeTravelAndFork:
    @pytest.mark.asyncio
    async def test_fork_creates_independent_thread(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)

        await agent.decompose("Sort a list", thread_id="src-thread")
        history = await agent.get_state_history("src-thread")
        # Pick a mid-history checkpoint
        cp_id = history[len(history) // 2]["checkpoint_id"]

        new_tid = await agent.fork("src-thread", cp_id, new_thread_id="fork-1")
        assert new_tid == "fork-1"

        # Forked thread has state
        forked_state = await agent.get_state("fork-1")
        assert forked_state["values"] is not None

    @pytest.mark.asyncio
    async def test_forked_state_matches_source_checkpoint(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)

        await agent.decompose("Sort a list", thread_id="src-2")
        history = await agent.get_state_history("src-2")
        cp_id = history[len(history) // 2]["checkpoint_id"]

        # Read source state at that checkpoint
        source_config = {
            "configurable": {
                "thread_id": "src-2",
                "checkpoint_id": cp_id,
            }
        }
        source_snapshot = await agent._graph.aget_state(source_config)

        await agent.fork("src-2", cp_id, new_thread_id="fork-2")
        forked = await agent.get_state("fork-2")

        # Key fields should match
        assert forked["values"]["goal"] == source_snapshot.values["goal"]
        assert len(forked["values"].get("nodes", [])) == len(
            source_snapshot.values.get("nodes", [])
        )

    @pytest.mark.asyncio
    async def test_original_unaffected_by_fork(self):
        saver = MemorySaver()
        agent = _build_agent(checkpointer=saver)

        await agent.decompose("Sort a list", thread_id="orig")
        orig_state = await agent.get_state("orig")
        orig_node_count = len(orig_state["values"].get("nodes", []))

        history = await agent.get_state_history("orig")
        cp_id = history[len(history) // 2]["checkpoint_id"]
        await agent.fork("orig", cp_id, new_thread_id="fork-3")

        # Original unchanged
        after_state = await agent.get_state("orig")
        assert len(after_state["values"].get("nodes", [])) == orig_node_count


# ---------------------------------------------------------------------------
# TestNoCheckpointerFallback
# ---------------------------------------------------------------------------

class TestNoCheckpointerFallback:
    @pytest.mark.asyncio
    async def test_agent_works_without_checkpointer(self):
        """Backward compat: checkpointer=None still runs fine."""
        agent = _build_agent(checkpointer=None)
        cdg = await agent.decompose("Sort a list")
        assert len(cdg.nodes) > 0
        assert cdg.metadata["goal"] == "Sort a list"


# ---------------------------------------------------------------------------
# TestHandoffValidation
# ---------------------------------------------------------------------------

def _valid_cdg() -> CDGExport:
    """CDG where all leaves are atomic with description + type_signature."""
    return CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="root",
                name="Root",
                description="Top-level goal",
                concept_type=ConceptType.DIVIDE_AND_CONQUER,
                status=NodeStatus.DECOMPOSED,
                children=["leaf1"],
                depth=0,
            ),
            AlgorithmicNode(
                node_id="leaf1",
                parent_id="root",
                name="Leaf",
                description="Sort the array",
                concept_type=ConceptType.SORTING,
                status=NodeStatus.ATOMIC,
                type_signature="list[int] -> list[int]",
                depth=1,
            ),
        ],
        edges=[
            DependencyEdge(
                source_id="root",
                target_id="leaf1",
                output_name="data",
                input_name="arr",
                source_type="list[int]",
                target_type="list[int]",
            ),
        ],
    )


class TestHandoffValidation:
    def test_valid_handoff_passes(self):
        cdg = _valid_cdg()
        issues = validate_handoff(cdg)
        assert issues == []

    def test_missing_type_signature_flagged(self):
        cdg = _valid_cdg()
        cdg.nodes[1].type_signature = ""
        issues = validate_handoff(cdg)
        assert any("type_signature" in i for i in issues)

    def test_missing_description_flagged(self):
        cdg = _valid_cdg()
        cdg.nodes[1].description = ""
        issues = validate_handoff(cdg)
        assert any("description" in i for i in issues)

    def test_non_atomic_leaf_flagged(self):
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="root",
                    name="Root",
                    description="Goal",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.DECOMPOSED,
                    children=["step"],
                    depth=0,
                ),
                AlgorithmicNode(
                    node_id="step",
                    parent_id="root",
                    name="Unfinished",
                    description="Not done",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.PENDING,
                    depth=1,
                ),
            ],
            edges=[],
        )
        issues = validate_handoff(cdg)
        assert any("not atomic" in i for i in issues)

    def test_strict_to_pdg_nodes_raises(self):
        cdg = _valid_cdg()
        cdg.nodes[1].type_signature = ""
        with pytest.raises(HandoffValidationError) as exc_info:
            to_pdg_nodes(cdg, strict=True)
        assert len(exc_info.value.issues) > 0

    def test_non_strict_to_pdg_nodes_skips_validation(self):
        cdg = _valid_cdg()
        # Valid CDG still works in non-strict mode
        pdg_nodes = to_pdg_nodes(cdg, strict=False)
        assert len(pdg_nodes) == 1

    def test_handoff_issues_convenience(self):
        cdg = _valid_cdg()
        assert cdg.handoff_issues() == []

        cdg.nodes[1].type_signature = ""
        issues = cdg.handoff_issues()
        assert len(issues) > 0
        assert any("type_signature" in i for i in issues)

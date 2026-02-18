"""Tests for Phase 2 semantic chunking (ageom.ingester.chunker)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from ageom.architect.models import ConceptType
from ageom.ingester.chunker import (
    ChunkerDeps,
    ChunkerState,
    build_chunker_graph,
    critic_validate,
    propose_macro_atoms,
)
from ageom.ingester.models import (
    MacroAtomSpec,
    MethodFact,
    ProposedMacroPlan,
    RawDataFlowGraph,
    ValidatedMacroPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dfg() -> RawDataFlowGraph:
    """Minimal data-flow graph with two methods and three attributes."""
    return RawDataFlowGraph(
        class_name="TestClass",
        source_code="class TestClass: ...",
        methods=[
            MethodFact(
                name="__init__",
                params=["data"],
                reads=[],
                writes=["raw", "result", "config"],
            ),
            MethodFact(
                name="process",
                params=[],
                reads=["raw"],
                writes=["result"],
            ),
        ],
        all_attributes={
            "raw": ["write:__init__", "read:process"],
            "result": ["write:__init__", "write:process"],
            "config": ["write:__init__"],
        },
    )


def _make_llm_response(method_names: list[str] | None = None) -> str:
    """Build a valid LLM JSON response for propose_macro_atoms."""
    if method_names is None:
        method_names = ["__init__", "process"]
    return json.dumps({
        "macro_atoms": [
            {
                "name": "Data Processor",
                "description": "Process raw data",
                "method_names": method_names,
                "inputs": [{"name": "data", "type_desc": "np.ndarray", "constraints": ""}],
                "outputs": [{"name": "result", "type_desc": "np.ndarray", "constraints": ""}],
                "config_params": [],
                "concept_type": "custom",
                "is_optional": False,
            }
        ],
        "edges": [],
    })


# ---------------------------------------------------------------------------
# Tests: propose_macro_atoms
# ---------------------------------------------------------------------------


class TestProposeMacroAtoms:
    @pytest.mark.asyncio
    async def test_parses_llm_response(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _make_llm_response()

        state: ChunkerState = {
            "raw_dfg": _make_dfg(),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=mock_llm)}}

        result = await propose_macro_atoms(state, config)
        plan = result["proposed_plan"]

        assert len(plan.macro_atoms) == 1
        assert plan.macro_atoms[0].name == "Data Processor"
        assert "process" in plan.macro_atoms[0].method_names

    @pytest.mark.asyncio
    async def test_handles_invalid_json(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = "not json at all"

        state: ChunkerState = {
            "raw_dfg": _make_dfg(),
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=mock_llm)}}

        result = await propose_macro_atoms(state, config)
        plan = result["proposed_plan"]
        assert len(plan.macro_atoms) == 0  # graceful fallback


# ---------------------------------------------------------------------------
# Tests: critic_validate
# ---------------------------------------------------------------------------


class TestCriticValidate:
    @pytest.mark.asyncio
    async def test_passes_when_all_attrs_covered(self):
        dfg = _make_dfg()
        plan = ProposedMacroPlan(
            macro_atoms=[
                MacroAtomSpec(
                    name="Processor",
                    method_names=["__init__", "process"],
                )
            ]
        )

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": plan,
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=AsyncMock())}}

        result = await critic_validate(state, config)
        assert result["critique_passed"] is True
        assert result["validated_plan"].all_attrs_accounted is True

    @pytest.mark.asyncio
    async def test_fails_when_attrs_missing(self):
        dfg = _make_dfg()
        # Only cover process method, not __init__
        plan = ProposedMacroPlan(
            macro_atoms=[
                MacroAtomSpec(
                    name="Processor",
                    method_names=["process"],  # misses __init__ writes
                )
            ]
        )

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": plan,
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=AsyncMock())}}

        result = await critic_validate(state, config)
        assert result["critique_passed"] is False
        assert "Missing" in result["critique_reason"]


# ---------------------------------------------------------------------------
# Tests: full chunker graph with mocked LLM
# ---------------------------------------------------------------------------


class TestChunkerGraph:
    @pytest.mark.asyncio
    async def test_end_to_end_pass(self):
        mock_llm = AsyncMock()
        # First call: propose_macro_atoms
        mock_llm.complete.side_effect = [
            _make_llm_response(["__init__", "process"]),
            # hoist_state (no cross-window attrs, so won't be called)
        ]

        dfg = _make_dfg()
        chunker = build_chunker_graph().compile()

        initial_state: dict = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=mock_llm)}}

        final = await chunker.ainvoke(initial_state, config=config)
        assert final.get("critique_passed", False) is True

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """First attempt misses attrs, second covers all."""
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            # First proposal: only covers 'process', misses __init__ attrs
            _make_llm_response(["process"]),
            # Retry proposal: covers both
            _make_llm_response(["__init__", "process"]),
        ]

        dfg = _make_dfg()
        chunker = build_chunker_graph().compile()

        initial_state: dict = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=mock_llm)}}

        final = await chunker.ainvoke(initial_state, config=config)
        assert final.get("critique_passed", False) is True
        assert mock_llm.complete.call_count >= 2

    @pytest.mark.asyncio
    async def test_max_retries_best_effort(self):
        """After max retries, exits with best-effort plan."""
        mock_llm = AsyncMock()
        # Always return incomplete coverage
        mock_llm.complete.return_value = _make_llm_response(["process"])

        dfg = _make_dfg()
        chunker = build_chunker_graph().compile()

        initial_state: dict = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": ValidatedMacroPlan(plan=ProposedMacroPlan()),
            "critique_passed": False,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": ChunkerDeps(llm=mock_llm)}}

        final = await chunker.ainvoke(initial_state, config=config)
        # Should exit after retries without passing
        assert final.get("critique_passed", False) is False
        assert final.get("retry_count", 0) >= 3

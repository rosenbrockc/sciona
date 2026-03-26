"""Tests for Phase 2 semantic chunking (sciona.ingester.chunker)."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sciona.ingester.chunker import (
    ChunkerDeps,
    ChunkerState,
    build_chunker_graph,
    critic_validate,
    propose_macro_atoms,
)
from sciona.ingester.models import (
    AttributeSemanticFact,
    FactProvenance,
    MacroAtomSpec,
    MethodFact,
    ParameterFact,
    ProposedMacroPlan,
    RawDataFlowGraph,
    ReturnFact,
    SourceSpan,
    ValidatedMacroPlan,
    runtime_macro_atoms,
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


def _make_simple_utility_dfg() -> RawDataFlowGraph:
    return RawDataFlowGraph(
        class_name="SignalUtility",
        methods=[
            MethodFact(
                name="normalize",
                params=["samples"],
                return_type="ndarray",
                docstring="Normalize raw samples before scoring.",
                reads=[],
                writes=[],
                source_code="def normalize(self, samples):\n    return samples / self.scale\n",
            ),
            MethodFact(
                name="score",
                params=["normalized"],
                return_type="float",
                docstring="Score the normalized signal.",
                reads=[],
                writes=[],
                source_code="def score(self, normalized):\n    return float(normalized.mean())\n",
            ),
            MethodFact(
                name="finalize",
                params=["score_value"],
                return_type="bool",
                docstring="Check the score against the threshold.",
                reads=[],
                writes=[],
                source_code="def finalize(self, score_value):\n    return score_value > self.threshold\n",
            ),
        ],
        all_attributes={},
        cross_window_attrs=[],
    )


def _make_internal_dispatch_dfg() -> RawDataFlowGraph:
    dfg = _make_simple_utility_dfg()
    dfg.methods[0].calls = ["score"]
    dfg.internal_call_graph = {"normalize": ["score"]}
    return dfg


def _make_inherited_dfg() -> RawDataFlowGraph:
    dfg = _make_simple_utility_dfg()
    dfg.opaque_base_classes = ["BaseUtility"]
    return dfg


def _make_semantic_ir_dfg() -> RawDataFlowGraph:
    prov = FactProvenance(
        rule_id="test",
        span=SourceSpan(file_path="semantic.py", line_start=1, line_end=1),
    )
    return RawDataFlowGraph(
        class_name="SemanticEstimator",
        methods=[
            MethodFact(
                name="__init__",
                params=["base_estimator"],
                writes=["base_estimator"],
                semantic_role="constructor",
                signature=[
                    ParameterFact(
                        name="base_estimator",
                        kind="positional_or_keyword",
                        provenance=prov,
                    )
                ],
                provenance=[prov],
            ),
            MethodFact(
                name="fit",
                params=["X", "y"],
                reads=["base_estimator"],
                writes=["calibrators", "is_fitted_"],
                semantic_role="fit_or_update",
                signature=[
                    ParameterFact(name="X", provenance=prov),
                    ParameterFact(name="y", provenance=prov),
                ],
                return_facts=[ReturnFact(kind="self", provenance=prov)],
                provenance=[prov],
            ),
            MethodFact(
                name="predict",
                params=["X"],
                reads=["calibrators"],
                writes=["last_prediction"],
                semantic_role="predict_or_transform",
                signature=[ParameterFact(name="X", provenance=prov)],
                return_facts=[ReturnFact(kind="call_result", provenance=prov)],
                provenance=[prov],
            ),
            MethodFact(
                name="get_calibrators",
                params=[],
                reads=["calibrators"],
                semantic_role="query_or_metadata",
                return_facts=[
                    ReturnFact(
                        kind="attribute",
                        referenced_attrs=["calibrators"],
                        provenance=prov,
                    )
                ],
                provenance=[prov],
            ),
            MethodFact(
                name="get_metadata_routing",
                params=[],
                semantic_role="query_or_metadata",
                return_facts=[ReturnFact(kind="constant", provenance=prov)],
                provenance=[prov],
            ),
        ],
        all_attributes={
            "base_estimator": ["write:__init__", "read:fit"],
            "calibrators": ["write:fit", "read:predict", "read:get_calibrators"],
            "is_fitted_": ["write:fit"],
            "last_prediction": ["write:predict"],
        },
        attribute_facts=[
            AttributeSemanticFact(
                attr_name="base_estimator",
                first_seen_in="__init__",
                read_methods=["fit"],
                write_methods=["__init__"],
                is_config=True,
                provenances=[prov],
            ),
            AttributeSemanticFact(
                attr_name="calibrators",
                first_seen_in="fit",
                read_methods=["predict", "get_calibrators"],
                write_methods=["fit"],
                is_fitted=True,
                provenances=[prov],
            ),
            AttributeSemanticFact(
                attr_name="is_fitted_",
                first_seen_in="fit",
                write_methods=["fit"],
                is_fitted=True,
                provenances=[prov],
            ),
            AttributeSemanticFact(
                attr_name="last_prediction",
                first_seen_in="predict",
                write_methods=["predict"],
                is_derived=True,
                provenances=[prov],
            ),
        ],
        config_attributes=["base_estimator"],
        fitted_attributes=["calibrators", "is_fitted_"],
        derived_attributes=["last_prediction"],
    )


def _make_non_python_semantic_ir_dfg() -> RawDataFlowGraph:
    prov = FactProvenance(
        rule_id="test",
        span=SourceSpan(file_path="integrator.rs", line_start=1, line_end=1),
    )
    return RawDataFlowGraph(
        class_name="Integrator",
        source_language="rust",
        methods=[
            MethodFact(
                name="step",
                params=["dt"],
                reads=["velocity"],
                writes=["position"],
                semantic_role="fit_or_update",
                signature=[ParameterFact(name="dt", annotation="f64", provenance=prov)],
                provenance=[prov],
            ),
            MethodFact(
                name="get_position",
                params=[],
                reads=["position"],
                semantic_role="query_or_metadata",
                return_facts=[
                    ReturnFact(
                        kind="attribute",
                        referenced_attrs=["position"],
                        provenance=prov,
                    )
                ],
                provenance=[prov],
            ),
        ],
        all_attributes={
            "velocity": ["read:step"],
            "position": ["write:step", "read:get_position"],
        },
        attribute_facts=[
            AttributeSemanticFact(
                attr_name="position",
                first_seen_in="step",
                read_methods=["get_position"],
                write_methods=["step"],
                is_fitted=True,
                provenances=[prov],
            ),
            AttributeSemanticFact(
                attr_name="velocity",
                first_seen_in="step",
                read_methods=["step"],
                is_config=True,
                provenances=[prov],
            ),
        ],
        config_attributes=["velocity"],
        fitted_attributes=["position"],
    )


def _make_llm_response(method_names: list[str] | None = None) -> str:
    """Build a valid LLM JSON response for propose_macro_atoms."""
    if method_names is None:
        method_names = ["__init__", "process"]
    return json.dumps(
        {
            "macro_atoms": [
                {
                    "name": "Data Processor",
                    "description": "Process raw data",
                    "method_names": method_names,
                    "inputs": [
                        {"name": "data", "type_desc": "np.ndarray", "constraints": ""}
                    ],
                    "outputs": [
                        {"name": "result", "type_desc": "np.ndarray", "constraints": ""}
                    ],
                    "config_params": [],
                    "concept_type": "custom",
                    "is_optional": False,
                }
            ],
            "edges": [],
        }
    )


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

    @pytest.mark.asyncio
    async def test_simple_class_chunks_by_public_method_without_llm(self):
        mock_llm = AsyncMock()
        state: ChunkerState = {
            "raw_dfg": _make_simple_utility_dfg(),
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

        assert [atom.name for atom in plan.macro_atoms] == [
            "Normalize",
            "Score",
            "Finalize",
        ]
        assert [atom.method_names for atom in plan.macro_atoms] == [
            ["normalize"],
            ["score"],
            ["finalize"],
        ]
        mock_llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_internal_dispatch_falls_back_to_llm_chunking(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _make_llm_response(["normalize", "score", "finalize"])
        state: ChunkerState = {
            "raw_dfg": _make_internal_dispatch_dfg(),
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

        assert result["proposed_plan"].macro_atoms[0].name == "Data Processor"
        mock_llm.complete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_complex_inheritance_falls_back_to_llm_chunking(self):
        mock_llm = AsyncMock()
        mock_llm.complete.return_value = _make_llm_response(["normalize", "score", "finalize"])
        state: ChunkerState = {
            "raw_dfg": _make_inherited_dfg(),
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

        assert result["proposed_plan"].macro_atoms[0].name == "Data Processor"
        mock_llm.complete.assert_awaited_once()


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

    @pytest.mark.asyncio
    async def test_validates_canonical_ir_and_role_bindings(self):
        dfg = _make_semantic_ir_dfg()
        plan = ProposedMacroPlan(
            macro_atoms=[
                MacroAtomSpec(
                    name="Estimator Fit",
                    method_names=["fit"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Predict",
                    method_names=["predict"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Metadata Routing",
                    method_names=["get_metadata_routing"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Get Calibrators",
                    method_names=["get_calibrators"],
                    inputs=[],
                    outputs=[],
                ),
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
        validated = result["validated_plan"]
        assert validated.ir_validated is True
        assert validated.plan.canonical_ir is not None
        operation_by_id = {
            op.operation_id: op for op in validated.plan.canonical_ir.operations
        }
        assert operation_by_id["estimator_fit"].role == "state_transition"
        assert operation_by_id["predict"].role == "predict"
        assert operation_by_id["get_calibrators"].role == "query"
        assert operation_by_id["metadata_routing"].role == "metadata"

        state_slots = {
            slot.slot_name: slot for slot in validated.plan.canonical_ir.state_slots
        }
        assert state_slots["base_estimator"].state_kind == "config"
        assert state_slots["calibrators"].state_kind == "fitted"
        assert state_slots["last_prediction"].state_kind == "derived"
        predict_outputs = operation_by_id["predict"].emitted_outputs
        assert len(predict_outputs) == 1
        assert predict_outputs[0].binding_kind == "return_value"
        query_outputs = operation_by_id["get_calibrators"].emitted_outputs
        assert len(query_outputs) == 1
        assert query_outputs[0].binding_kind == "attribute_read"
        metadata_outputs = operation_by_id["metadata_routing"].emitted_outputs
        assert len(metadata_outputs) == 1
        assert metadata_outputs[0].binding_kind == "metadata_object"

    @pytest.mark.asyncio
    async def test_canonical_ir_drives_compat_outputs_when_legacy_exports_are_sparse(self):
        dfg = _make_semantic_ir_dfg()
        plan = ProposedMacroPlan(
            macro_atoms=[
                MacroAtomSpec(
                    name="Estimator Fit",
                    method_names=["fit"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Predict",
                    method_names=["predict"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Metadata Routing",
                    method_names=["get_metadata_routing"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Get Calibrators",
                    method_names=["get_calibrators"],
                    inputs=[],
                    outputs=[],
                ),
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
        validated_plan = result["validated_plan"].plan
        assert all(not atom.outputs for atom in validated_plan.macro_atoms)
        ir = validated_plan.canonical_ir
        assert ir is not None
        operation_by_id = {op.operation_id: op for op in ir.operations}
        assert operation_by_id["predict"].emitted_outputs[0].output_name == "result"
        assert operation_by_id["metadata_routing"].emitted_outputs[0].output_name == "result"
        assert operation_by_id["get_calibrators"].emitted_outputs[0].output_name == "calibrators"

        adapted_atoms = runtime_macro_atoms(validated_plan)

        predict_atom = next(atom for atom in adapted_atoms if atom.name == "Predict")
        metadata_atom = next(
            atom for atom in adapted_atoms if atom.name == "Metadata Routing"
        )
        query_atom = next(
            atom for atom in adapted_atoms if atom.name == "Get Calibrators"
        )
        assert predict_atom.outputs[0].name == "result"
        assert metadata_atom.outputs[0].name == "result"
        assert query_atom.outputs[0].name == "calibrators"
        assert all(not atom.outputs for atom in validated_plan.macro_atoms)

    @pytest.mark.asyncio
    async def test_builds_canonical_ir_for_non_python_semantic_facts(self):
        dfg = _make_non_python_semantic_ir_dfg()
        plan = ProposedMacroPlan(
            macro_atoms=[
                MacroAtomSpec(
                    name="Integrator Step",
                    method_names=["step"],
                    inputs=[],
                    outputs=[],
                ),
                MacroAtomSpec(
                    name="Get Position",
                    method_names=["get_position"],
                    inputs=[],
                    outputs=[],
                ),
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
        ir = result["validated_plan"].plan.canonical_ir
        assert ir is not None
        assert ir.source_language == "rust"
        operation_by_id = {op.operation_id: op for op in ir.operations}
        assert operation_by_id["integrator_step"].role == "state_transition"
        assert operation_by_id["get_position"].role == "query"
        assert operation_by_id["get_position"].emitted_outputs[0].binding_kind == "attribute_read"


# ---------------------------------------------------------------------------
# Tests: full chunker graph with mocked LLM
# ---------------------------------------------------------------------------


class TestChunkerGraph:
    @pytest.mark.asyncio
    async def test_end_to_end_pass(self):
        mock_llm = AsyncMock()
        mock_llm.complete.side_effect = [
            _make_llm_response(["__init__", "process"]),
            json.dumps(
                {
                    "abstract_name": "Data Processor",
                    "conceptual_transform": "Process data",
                    "abstract_inputs": [],
                    "abstract_outputs": [],
                    "algorithmic_properties": [],
                    "cross_disciplinary_applications": [],
                }
            ),
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
            json.dumps(
                {
                    "abstract_name": "Data Processor",
                    "conceptual_transform": "Process data",
                    "abstract_inputs": [],
                    "abstract_outputs": [],
                    "algorithmic_properties": [],
                    "cross_disciplinary_applications": [],
                }
            ),
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

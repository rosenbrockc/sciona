"""Tests for recursive chunking with ingester_max_depth."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock

import pytest

from sciona.architect.models import ConceptType, IOSpec, NodeStatus
from sciona.config import AgeomConfig
from sciona.ingester.chunker import (
    ChunkerDeps,
    ChunkerState,
    decompose_complex_atoms,
    is_atom_complex,
)
from sciona.ingester.emitter import build_cdg_export
from sciona.ingester.models import (
    IngestIRPlan,
    MacroAtomSpec,
    MethodFact,
    MethodBinding,
    OperationSpec,
    OutputBindingSpec,
    ProposedMacroPlan,
    RawDataFlowGraph,
    StateEffectSpec,
    ValidatedMacroPlan,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_dfg(
    source_lines: int = 50,
    num_calls: int = 0,
    has_not_implemented: bool = False,
) -> RawDataFlowGraph:
    """Build a DFG with one method of controllable complexity."""
    body = "\n".join(f"    x = x + {i}" for i in range(source_lines))
    if has_not_implemented:
        body = '    raise NotImplementedError("TODO")'
    calls = [f"_helper_{i}" for i in range(num_calls)]
    helper_methods = [
        MethodFact(name=f"_helper_{i}", params=[], source_code="pass")
        for i in range(num_calls)
    ]
    return RawDataFlowGraph(
        class_name="BigClass",
        methods=[
            MethodFact(
                name="run_pipeline",
                params=["data"],
                reads=["raw"],
                writes=["result"],
                calls=calls,
                source_code=f"def run_pipeline(self, data):\n{body}",
            ),
            *helper_methods,
        ],
        all_attributes={
            "raw": ["read:run_pipeline"],
            "result": ["write:run_pipeline"],
        },
    )


def _make_atom(name: str = "Run Pipeline", method_names: list[str] | None = None) -> MacroAtomSpec:
    return MacroAtomSpec(
        name=name,
        description="A complex pipeline step",
        method_names=method_names or ["run_pipeline"],
        inputs=[IOSpec(name="data", type_desc="np.ndarray")],
        outputs=[IOSpec(name="result", type_desc="np.ndarray")],
    )


def _make_validated_plan(atoms: list[MacroAtomSpec]) -> ValidatedMacroPlan:
    return ValidatedMacroPlan(
        plan=ProposedMacroPlan(macro_atoms=atoms),
        all_attrs_accounted=True,
    )


def _make_ir_plan(
    operation: OperationSpec,
    atom: MacroAtomSpec,
) -> ValidatedMacroPlan:
    return ValidatedMacroPlan(
        plan=ProposedMacroPlan(
            macro_atoms=[atom],
            canonical_ir=IngestIRPlan(
                subject_name="BigClass",
                operations=[operation],
            ),
        ),
        all_attrs_accounted=True,
    )


def _decompose_response(num_sub_atoms: int = 3) -> str:
    """Build a mock LLM decomposition JSON response."""
    sub_atoms = []
    for i in range(num_sub_atoms):
        sub_atoms.append(
            {
                "name": f"Sub Step {i+1}",
                "description": f"Sub-step {i+1} of the pipeline",
                "inputs": [{"name": "data", "type_desc": "np.ndarray", "constraints": ""}],
                "outputs": [{"name": "result", "type_desc": "np.ndarray", "constraints": ""}],
                "concept_type": "custom",
            }
        )
    edges = []
    for i in range(num_sub_atoms - 1):
        edges.append(
            {
                "source_id": f"Sub Step {i+1}",
                "target_id": f"Sub Step {i+2}",
                "output_name": "result",
                "input_name": "data",
                "source_type": "np.ndarray",
                "target_type": "np.ndarray",
            }
        )
    return json.dumps({"sub_atoms": sub_atoms, "edges": edges})


# ---------------------------------------------------------------------------
# TestComplexityHeuristic
# ---------------------------------------------------------------------------


class TestComplexityHeuristic:
    """Verify the deterministic is_atom_complex heuristic."""

    def test_short_method_not_complex(self):
        dfg = _make_dfg(source_lines=10)
        atom = _make_atom()
        assert not is_atom_complex(atom, dfg, line_threshold=30)

    def test_long_method_is_complex(self):
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        assert is_atom_complex(atom, dfg, line_threshold=30)

    def test_many_calls_is_complex(self):
        dfg = _make_dfg(source_lines=10, num_calls=4)
        atom = _make_atom()
        assert is_atom_complex(atom, dfg, line_threshold=30)

    def test_two_calls_not_complex(self):
        dfg = _make_dfg(source_lines=10, num_calls=2)
        atom = _make_atom()
        assert not is_atom_complex(atom, dfg, line_threshold=30)

    def test_not_implemented_is_complex(self):
        dfg = _make_dfg(source_lines=2, has_not_implemented=True)
        atom = _make_atom()
        assert is_atom_complex(atom, dfg, line_threshold=30)

    def test_source_lines_field_honoured(self):
        dfg = _make_dfg(source_lines=5)
        atom = _make_atom()
        atom.source_lines = 50
        assert is_atom_complex(atom, dfg, line_threshold=30)

    def test_custom_threshold(self):
        dfg = _make_dfg(source_lines=20)
        atom = _make_atom()
        assert not is_atom_complex(atom, dfg, line_threshold=30)
        assert is_atom_complex(atom, dfg, line_threshold=15)


class TestIngesterDepthDefaults:
    def test_config_default_depth_is_12(self):
        assert AgeomConfig.model_fields["ingester_max_depth"].default == 12


# ---------------------------------------------------------------------------
# TestDecomposeComplexAtoms
# ---------------------------------------------------------------------------


class TestDecomposeComplexAtoms:
    """Mock LLM, verify sub-atom creation at depth 2."""

    @pytest.mark.asyncio
    async def test_decompose_creates_children(self):
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        plan = _make_validated_plan([atom])

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(3))
        # select_llm returns the same mock (no router)
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        updated_plan = result["validated_plan"]
        decomposed_atom = updated_plan.plan.macro_atoms[0]

        assert len(decomposed_atom.children) == 3
        assert decomposed_atom.children[0].name == "Sub Step 1"

    @pytest.mark.asyncio
    async def test_decompose_materializes_runtime_atoms_from_canonical_ir(self):
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        operation = OperationSpec(
            operation_id="run_pipeline",
            display_name="Run Pipeline",
            role="state_transition",
            method_bindings=[MethodBinding(method_name="run_pipeline")],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="return_value",
                    source_method="run_pipeline",
                )
            ],
            state_effects=[StateEffectSpec(slot_name="result", effect_kind="update")],
        )
        plan = ValidatedMacroPlan(
            plan=ProposedMacroPlan(
                macro_atoms=[],
                canonical_ir=IngestIRPlan(
                    subject_name="BigClass",
                    operations=[operation],
                ),
            ),
            all_attrs_accounted=True,
        )

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(2))
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)

        updated_plan = result["validated_plan"].plan
        decomposed_atom = updated_plan.macro_atoms[0]
        assert updated_plan.canonical_ir is not None
        assert decomposed_atom.name == "Run Pipeline"
        assert len(decomposed_atom.children) == 2
        assert decomposed_atom.children[0].depth == 2

    @pytest.mark.asyncio
    async def test_decompose_skips_operationless_legacy_fallback_for_top_level_atoms(self):
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        operation = OperationSpec(
            operation_id="canonical_pipeline_op",
            display_name="Run Pipeline",
            role="state_transition",
            method_bindings=[MethodBinding(method_name="run_pipeline")],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="return_value",
                    source_method="run_pipeline",
                )
            ],
            state_effects=[StateEffectSpec(slot_name="result", effect_kind="update")],
        )
        plan = ValidatedMacroPlan(
            plan=ProposedMacroPlan(
                macro_atoms=[atom],
                canonical_ir=IngestIRPlan(
                    subject_name="BigClass",
                    operations=[operation],
                ),
            ),
            all_attrs_accounted=True,
        )

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(2))
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        updated_plan = result["validated_plan"]

        assert updated_plan.plan.macro_atoms[0].children == []
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_simple_atom_untouched(self):
        dfg = _make_dfg(source_lines=10)
        atom = _make_atom()
        plan = _make_validated_plan([atom])

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(3))
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        updated_plan = result["validated_plan"]
        assert len(updated_plan.plan.macro_atoms[0].children) == 0

    @pytest.mark.asyncio
    async def test_metadata_operation_stays_atomic_without_llm(self):
        dfg = _make_dfg(source_lines=50)
        dfg.methods[0].semantic_role = "query_or_metadata"
        atom = _make_atom(name="Metadata View")
        operation = OperationSpec(
            operation_id="metadata_view",
            display_name="Metadata View",
            role="metadata",
            method_bindings=[MethodBinding(method_name="run_pipeline")],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="metadata_object",
                    source_method="run_pipeline",
                )
            ],
        )
        plan = _make_ir_plan(operation, atom)

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(3))
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        updated_plan = result["validated_plan"]

        assert updated_plan.plan.macro_atoms[0].children == []
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_multi_method_state_transition_decomposes_deterministically(self):
        dfg = RawDataFlowGraph(
            class_name="BigClass",
            methods=[
                MethodFact(
                    name="fit_prepare",
                    params=["data"],
                    reads=[],
                    writes=["prepared"],
                    semantic_role="fit_or_update",
                    source_code="def fit_prepare(self, data):\n    self.prepared = data\n",
                ),
                MethodFact(
                    name="fit_update",
                    params=["data"],
                    reads=["prepared"],
                    writes=["result"],
                    semantic_role="fit_or_update",
                    source_code="def fit_update(self, data):\n    self.result = self.prepared\n",
                ),
            ],
            all_attributes={
                "prepared": ["write:fit_prepare", "read:fit_update"],
                "result": ["write:fit_update"],
            },
        )
        atom = _make_atom(name="Fit Pipeline", method_names=["fit_prepare", "fit_update"])
        operation = OperationSpec(
            operation_id="fit_pipeline",
            display_name="Fit Pipeline",
            role="state_transition",
            method_bindings=[
                MethodBinding(method_name="fit_prepare"),
                MethodBinding(method_name="fit_update"),
            ],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="attribute_read",
                    source_method="fit_update",
                    source_attr="result",
                )
            ],
            state_effects=[
                StateEffectSpec(
                    slot_name="prepared",
                    effect_kind="update",
                    source_method="fit_prepare",
                ),
                StateEffectSpec(
                    slot_name="result",
                    effect_kind="update",
                    source_method="fit_update",
                ),
            ],
        )
        plan = _make_ir_plan(operation, atom)

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(2))
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        decomposed_atom = result["validated_plan"].plan.macro_atoms[0]

        assert [child.name for child in decomposed_atom.children] == [
            "Fit Prepare",
            "Fit Update",
        ]
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_blocked_unknown_skips_llm(self):
        dfg = _make_dfg(source_lines=80)
        atom = _make_atom(name="Blocked Step")
        operation = OperationSpec(
            operation_id="blocked_step",
            display_name="Blocked Step",
            role="state_transition",
            method_bindings=[MethodBinding(method_name="run_pipeline")],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="unknown",
                    source_method="run_pipeline",
                )
            ],
        )
        plan = _make_ir_plan(operation, atom)

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(3))
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)

        assert result["validated_plan"].plan.macro_atoms[0].children == []
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_llm_decomposition_rejects_invented_outputs(self):
        dfg = _make_dfg(source_lines=80)
        atom = _make_atom(name="Fallback Step")
        operation = OperationSpec(
            operation_id="fallback_step",
            display_name="Fallback Step",
            role="unknown",
            method_bindings=[MethodBinding(method_name="run_pipeline")],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="return_value",
                    source_method="run_pipeline",
                )
            ],
        )
        plan = _make_ir_plan(operation, atom)

        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "sub_atoms": [
                        {
                            "name": "Invented Child",
                            "description": "bad",
                            "inputs": [
                                {
                                    "name": "data",
                                    "type_desc": "np.ndarray",
                                    "constraints": "",
                                }
                            ],
                            "outputs": [
                                {
                                    "name": "invented",
                                    "type_desc": "np.ndarray",
                                    "constraints": "",
                                }
                            ],
                            "concept_type": "custom",
                        }
                    ],
                    "edges": [],
                }
            )
        )
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)

        assert result["validated_plan"].plan.macro_atoms[0].children == []
        llm.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_decompose_dedupes_duplicate_children(self):
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        plan = _make_validated_plan([atom])

        llm = AsyncMock()
        llm.complete = AsyncMock(
            return_value=json.dumps(
                {
                    "sub_atoms": [
                        {
                            "name": "Sub Step 1",
                            "description": "first",
                            "inputs": [{"name": "data", "type_desc": "np.ndarray", "constraints": ""}],
                            "outputs": [{"name": "result", "type_desc": "np.ndarray", "constraints": ""}],
                            "concept_type": "custom",
                        },
                        {
                            "name": "Sub Step 1",
                            "description": "duplicate with richer docs",
                            "inputs": [{"name": "data", "type_desc": "np.ndarray", "constraints": "non-empty"}],
                            "outputs": [{"name": "result", "type_desc": "np.ndarray", "constraints": "non-empty"}],
                            "concept_type": "custom",
                        },
                    ],
                    "edges": [],
                }
            )
        )
        deps = ChunkerDeps(llm=llm, max_depth=3, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        decomposed_atom = result["validated_plan"].plan.macro_atoms[0]

        assert [child.name for child in decomposed_atom.children] == ["Sub Step 1"]
        assert decomposed_atom.children[0].description == "duplicate with richer docs"


# ---------------------------------------------------------------------------
# TestMaxDepthRespected
# ---------------------------------------------------------------------------


class TestMaxDepthRespected:
    """Verify recursion stops at configured max_depth."""

    @pytest.mark.asyncio
    async def test_depth_1_skips_decomposition(self):
        """max_depth=1 should return plan unchanged (no LLM calls)."""
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        plan = _make_validated_plan([atom])

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(3))
        deps = ChunkerDeps(llm=llm, max_depth=1, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        # Plan should be unchanged
        assert result["validated_plan"] is plan
        # LLM should NOT have been called
        llm.complete.assert_not_called()

    @pytest.mark.asyncio
    async def test_depth_2_decomposes_once(self):
        """max_depth=2 → one level of decomposition, no further recursion."""
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        plan = _make_validated_plan([atom])

        llm = AsyncMock()
        llm.complete = AsyncMock(return_value=_decompose_response(2))
        deps = ChunkerDeps(llm=llm, max_depth=2, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        updated = result["validated_plan"]
        decomposed = updated.plan.macro_atoms[0]
        assert len(decomposed.children) == 2
        # Children should have no further children (depth 2 = max)
        for child in decomposed.children:
            assert len(child.children) == 0


# ---------------------------------------------------------------------------
# TestDepth1Unchanged
# ---------------------------------------------------------------------------


class TestDepth1Unchanged:
    """Verify max_depth=1 produces identical output to current behavior."""

    @pytest.mark.asyncio
    async def test_identical_output(self):
        dfg = _make_dfg(source_lines=50)
        atom = _make_atom()
        original_plan = _make_validated_plan([atom])

        llm = AsyncMock()
        deps = ChunkerDeps(llm=llm, max_depth=1, line_threshold=30)

        state: ChunkerState = {
            "raw_dfg": dfg,
            "proposed_plan": ProposedMacroPlan(),
            "validated_plan": original_plan,
            "critique_passed": True,
            "critique_reason": "",
            "retry_count": 0,
            "missing_attrs": [],
            "done": False,
        }
        config = {"configurable": {"deps": deps}}

        result = await decompose_complex_atoms(state, config)
        # Should be the exact same object
        assert result["validated_plan"] is original_plan
        assert result["validated_plan"].plan.macro_atoms[0].children == []


# ---------------------------------------------------------------------------
# TestCDGDepthPropagation
# ---------------------------------------------------------------------------


class TestCDGDepthPropagation:
    """Verify build_cdg_export emits correct depth values for nested atoms."""

    def test_flat_cdg_depths(self):
        """No children → root at 0, children at 1."""
        atom_a = MacroAtomSpec(name="Step A", description="a")
        atom_b = MacroAtomSpec(name="Step B", description="b")
        plan = _make_validated_plan([atom_a, atom_b])

        cdg = build_cdg_export(plan, "TestClass")

        root = cdg.nodes[0]
        assert root.depth == 0
        assert root.status == NodeStatus.DECOMPOSED

        for child in cdg.nodes[1:]:
            assert child.depth == 1
            assert child.status == NodeStatus.ATOMIC

    def test_nested_cdg_depths(self):
        """Atoms with children → parent DECOMPOSED, children at depth+1."""
        sub1 = MacroAtomSpec(name="Sub 1", description="s1", depth=2)
        sub2 = MacroAtomSpec(name="Sub 2", description="s2", depth=2)
        parent = MacroAtomSpec(
            name="Big Step",
            description="big",
            children=[sub1, sub2],
        )
        plan = _make_validated_plan([parent])

        cdg = build_cdg_export(plan, "TestClass")

        # root (depth=0), parent (depth=1), sub1 (depth=2), sub2 (depth=2)
        assert len(cdg.nodes) == 4

        root = cdg.nodes[0]
        assert root.depth == 0
        assert root.status == NodeStatus.DECOMPOSED

        parent_node = next(n for n in cdg.nodes if n.name == "Big Step")
        assert parent_node.depth == 1
        assert parent_node.status == NodeStatus.DECOMPOSED
        assert len(parent_node.children) == 2

        sub_nodes = [n for n in cdg.nodes if n.name.startswith("Sub")]
        for sn in sub_nodes:
            assert sn.depth == 2
            assert sn.status == NodeStatus.ATOMIC

    def test_three_level_nesting(self):
        """Three levels of nesting: root → parent → child → grandchild."""
        grandchild = MacroAtomSpec(name="Leaf", description="gc", depth=3)
        child = MacroAtomSpec(
            name="Middle",
            description="mid",
            children=[grandchild],
            depth=2,
        )
        parent = MacroAtomSpec(
            name="Top",
            description="top",
            children=[child],
        )
        plan = _make_validated_plan([parent])

        cdg = build_cdg_export(plan, "TestClass")

        # root + top + middle + leaf = 4 nodes
        assert len(cdg.nodes) == 4

        leaf = next(n for n in cdg.nodes if n.name == "Leaf")
        assert leaf.depth == 3
        assert leaf.status == NodeStatus.ATOMIC

        middle = next(n for n in cdg.nodes if n.name == "Middle")
        assert middle.depth == 2
        assert middle.status == NodeStatus.DECOMPOSED

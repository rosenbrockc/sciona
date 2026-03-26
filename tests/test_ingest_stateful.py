"""Integration test: ingest a stateful rolling-state class.

Exercises the full stateful pipeline: AST extraction of cross-window
state, stateful wrapper generation with inject/run/extract pattern,
deterministic state edge computation, and end-to-end bundle assembly.

Mock class: RollingAverager — maintains a rolling buffer and count
across calls, with two methods that share state via self.buffer
and self.count.
"""

from __future__ import annotations

import ast
import json
import textwrap
from unittest.mock import AsyncMock

import pytest

from sciona.architect.models import ConceptType, DependencyEdge, IOSpec
from sciona.ingester.chunker import _compute_state_edges
from sciona.ingester.emitter import (
    emit_ingestion_bundle,
    generate_ghost_witnesses,
    generate_stateful_wrappers,
)
from sciona.ingester.extractor import _compute_cross_window_attrs, extract_data_flow
from sciona.ingester.graph import IngesterAgent
from sciona.ingester.models import (
    IngestIRPlan,
    IngestPlanGraph,
    MacroAtomSpec,
    MethodBinding,
    OperationSpec,
    OutputBindingSpec,
    ParameterFact,
    PlannedOperationGroup,
    ProposedMacroPlan,
    StateEffectSpec,
    StateModelSpec,
    StateSlotSpec,
    ValidatedMacroPlan,
)

# ---------------------------------------------------------------------------
# Mock class source
# ---------------------------------------------------------------------------

ROLLING_AVERAGER_SOURCE = textwrap.dedent("""\
    class RollingAverager:
        def __init__(self, window_size: int = 5):
            self.window_size = window_size
            self.buffer: list = []
            self.count: int = 0
            self.result: float = 0.0

        def add_sample(self, value: float) -> None:
            self.buffer.append(value)
            if len(self.buffer) > self.window_size:
                self.buffer = self.buffer[-self.window_size:]
            self.count += 1

        def compute_average(self) -> float:
            if not self.buffer:
                self.result = 0.0
            else:
                self.result = sum(self.buffer) / len(self.buffer)
            return self.result
""")


# ---------------------------------------------------------------------------
# Mock LLM responses (deterministic, like BioSPPy test pattern)
# ---------------------------------------------------------------------------

_CHUNK_RESPONSE = json.dumps(
    {
        "macro_atoms": [
            {
                "name": "Sample Accumulator",
                "description": "Accumulate sample values into a rolling buffer",
                "method_names": ["add_sample"],
                "inputs": [
                    {
                        "name": "value",
                        "type_desc": "float",
                        "constraints": "numeric sample value",
                    },
                ],
                "outputs": [],
                "config_params": ["window_size"],
                "concept_type": "custom",
                "is_optional": False,
            },
            {
                "name": "Average Computer",
                "description": "Compute the rolling average from the buffer",
                "method_names": ["compute_average"],
                "inputs": [],
                "outputs": [
                    {
                        "name": "result",
                        "type_desc": "float",
                        "constraints": "rolling average value",
                    },
                ],
                "config_params": [],
                "concept_type": "custom",
                "is_optional": False,
            },
        ],
        "edges": [
            {
                "source_id": "sample_accumulator",
                "target_id": "average_computer",
                "output_name": "buffer",
                "input_name": "buffer",
                "source_type": "list",
                "target_type": "list",
            },
        ],
    }
)

_HOIST_RESPONSE = json.dumps(
    {
        "state_models": [
            {
                "model_name": "RollingAveragerState",
                "fields": [
                    ["buffer", "list"],
                    ["count", "int"],
                ],
                "source_attrs": ["buffer", "count"],
                "docstring": "Cross-window state for the rolling averager.",
            },
        ],
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_agent() -> tuple[IngesterAgent, AsyncMock]:
    """Build an IngesterAgent with a mocked LLM returning known-correct data."""
    mock_llm = AsyncMock()
    mock_llm.complete.side_effect = [_CHUNK_RESPONSE, _HOIST_RESPONSE]
    agent = IngesterAgent(llm=mock_llm)
    return agent, mock_llm


def _make_stateful_plan() -> ValidatedMacroPlan:
    """Build a ValidatedMacroPlan matching the mock LLM responses."""
    atoms = [
        MacroAtomSpec(
            name="Sample Accumulator",
            description="Accumulate sample values into a rolling buffer",
            method_names=["add_sample"],
            inputs=[
                IOSpec(
                    name="value", type_desc="float", constraints="numeric sample value"
                )
            ],
            outputs=[],
            config_params=["window_size"],
            concept_type=ConceptType.CUSTOM,
        ),
        MacroAtomSpec(
            name="Average Computer",
            description="Compute the rolling average from the buffer",
            method_names=["compute_average"],
            inputs=[],
            outputs=[
                IOSpec(
                    name="result",
                    type_desc="float",
                    constraints="rolling average value",
                )
            ],
            concept_type=ConceptType.CUSTOM,
        ),
    ]
    state_models = [
        StateModelSpec(
            model_name="RollingAveragerState",
            fields=[("buffer", "list"), ("count", "int")],
            source_attrs=["buffer", "count"],
            docstring="Cross-window state for the rolling averager.",
        ),
    ]
    edges = [
        DependencyEdge(
            source_id="sample_accumulator",
            target_id="average_computer",
            output_name="buffer",
            input_name="buffer",
            source_type="list",
            target_type="list",
        ),
    ]
    plan = ProposedMacroPlan(
        macro_atoms=atoms,
        state_models=state_models,
        edge_definitions=edges,
    )
    return ValidatedMacroPlan(
        plan=plan,
        all_attrs_accounted=True,
        coverage_report="All attributes accounted for.",
    )


def _make_canonical_stateful_plan() -> ValidatedMacroPlan:
    atom = MacroAtomSpec(
        name="Average Computer",
        description="Compute a rolling summary from retained state",
        method_names=["compute_average"],
        inputs=[IOSpec(name="window_size", type_desc="int", constraints="")],
        outputs=[IOSpec(name="result", type_desc="float", constraints="")],
        concept_type=ConceptType.CUSTOM,
    )
    state_model = StateModelSpec(
        model_name="RollingAveragerState",
        fields=[("buffer", "list"), ("count", "int"), ("unused", "str")],
        source_attrs=["buffer", "count", "unused"],
        docstring="Cross-window state for the rolling averager.",
    )
    operation = OperationSpec(
        operation_id="average_computer",
        display_name="Average Computer",
        role="query",
        method_bindings=[
            MethodBinding(
                method_name="compute_average",
                signature=[ParameterFact(name="self")],
            )
        ],
        direct_inputs=[],
        required_state_slots=["buffer", "window_size"],
        emitted_outputs=[
            OutputBindingSpec(
                output_name="result",
                type_desc="float",
                binding_kind="attribute_read",
                source_method="compute_average",
                source_attr="result",
            )
        ],
        state_effects=[
            StateEffectSpec(
                slot_name="count",
                effect_kind="update",
                source_method="compute_average",
            )
        ],
        concept_type=ConceptType.CUSTOM,
    )
    plan = ProposedMacroPlan(
        macro_atoms=[atom],
        state_models=[state_model],
        canonical_ir=IngestIRPlan(
            subject_name="RollingAverager",
            operations=[operation],
            state_slots=[
                StateSlotSpec(slot_name="buffer", state_kind="fitted", type_desc="list"),
                StateSlotSpec(slot_name="count", state_kind="fitted", type_desc="int"),
                StateSlotSpec(slot_name="unused", state_kind="derived", type_desc="str"),
                StateSlotSpec(slot_name="window_size", state_kind="config", type_desc="int"),
            ],
        ),
    )
    return ValidatedMacroPlan(plan=plan, all_attrs_accounted=True)


# ---------------------------------------------------------------------------
# Tests: Phase 1 — deterministic AST extraction of stateful class
# ---------------------------------------------------------------------------


class TestPhase1StatefulExtraction:
    """Verify the deterministic AST extractor captures rolling-state patterns."""

    @pytest.fixture
    def ra_source(self, tmp_path) -> str:
        src = tmp_path / "rolling_averager.py"
        src.write_text(ROLLING_AVERAGER_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_finds_buffer_reads_and_writes(self, ra_source):
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        add = next(m for m in dfg.methods if m.name == "add_sample")
        assert "buffer" in add.reads
        assert "buffer" in add.writes

    @pytest.mark.asyncio
    async def test_finds_count_reads_and_writes(self, ra_source):
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        add = next(m for m in dfg.methods if m.name == "add_sample")
        assert "count" in add.writes

    @pytest.mark.asyncio
    async def test_cross_window_detected(self, ra_source):
        """buffer and count are read+written in non-init methods => cross-window."""
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        assert "buffer" in dfg.cross_window_attrs

    @pytest.mark.asyncio
    async def test_init_chain_correct(self, ra_source):
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        assert set(dfg.init_chain) == {"window_size", "buffer", "count", "result"}

    @pytest.mark.asyncio
    async def test_compute_cross_window_attrs_directly(self, ra_source):
        """Test the _compute_cross_window_attrs helper directly."""
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        cross = _compute_cross_window_attrs(dfg.methods)
        assert "buffer" in cross


# ---------------------------------------------------------------------------
# Tests: Stateful wrapper generation
# ---------------------------------------------------------------------------


class TestStatefulWrapperGeneration:
    """Verify generate_stateful_wrappers produces correct inject/run/extract code."""

    def test_wrapper_has_state_param(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        assert "state: RollingAveragerState" in source

    def test_wrapper_return_type_is_tuple(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        # The average_computer wrapper should return tuple[float, RollingAveragerState]
        assert "RollingAveragerState]" in source

    def test_wrapper_contains_dunder_new(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        assert "RollingAverager.__new__(RollingAverager)" in source

    def test_wrapper_contains_model_copy(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        assert "model_copy" in source

    def test_wrapper_injects_all_state_fields(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        assert "obj.buffer = state.buffer" in source
        assert "obj.count = state.count" in source

    def test_wrapper_extracts_all_state_fields(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        assert '"buffer": obj.buffer' in source
        assert '"count": obj.count' in source

    def test_wrapper_is_valid_python(self):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
        )
        ast.parse(source)

    def test_wrapper_imports_generated_modules_and_source_class(self, tmp_path):
        plan = _make_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source_path = tmp_path / "rolling_averager.py"
        source_path.write_text(ROLLING_AVERAGER_SOURCE)
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
            source_file=str(source_path),
        )
        assert "from state_models import RollingAveragerState" in source
        assert (
            "from witnesses import witness_average_computer, witness_sample_accumulator"
            in source
        )
        assert "_SCIONA_SOURCE_FILE" in source
        assert 'RollingAverager: Any = getattr(_SCIONA_SOURCE_MODULE, "RollingAverager")' in source

    def test_canonical_wrapper_injects_and_extracts_only_declared_slots(self):
        plan = _make_canonical_stateful_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
            source_file="rolling_averager.py",
            plan=plan,
        )

        assert "window_size: int, state: RollingAveragerState" in source
        assert "obj.buffer = state.buffer" in source
        assert "obj.window_size = window_size" in source
        assert "obj.unused = state.unused" not in source
        assert '"count": obj.count' in source
        assert '"buffer": obj.buffer' not in source
        assert 'return obj.result, new_state' in source

    def test_canonical_query_wrapper_preserves_state_when_no_effects(self):
        atom = MacroAtomSpec(
            name="Metadata View",
            description="Return a metadata snapshot without mutation",
            method_names=["metadata_view"],
            inputs=[],
            outputs=[IOSpec(name="metadata", type_desc="dict[str, Any]", constraints="")],
            concept_type=ConceptType.CUSTOM,
        )
        state_model = StateModelSpec(
            model_name="RollingAveragerState",
            fields=[("buffer", "list"), ("count", "int")],
        )
        operation = OperationSpec(
            operation_id="metadata_view",
            display_name="Metadata View",
            role="metadata",
            method_bindings=[MethodBinding(method_name="metadata_view", signature=[ParameterFact(name="self")])],
            required_state_slots=["buffer"],
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="metadata",
                    type_desc="dict[str, Any]",
                    binding_kind="metadata_object",
                    source_method="metadata_view",
                )
            ],
        )
        plan = ValidatedMacroPlan(
            plan=ProposedMacroPlan(
                macro_atoms=[atom],
                state_models=[state_model],
                canonical_ir=IngestIRPlan(
                    subject_name="RollingAverager",
                    operations=[operation],
                    state_slots=[StateSlotSpec(slot_name="buffer", state_kind="fitted", type_desc="list")],
                ),
            ),
            all_attrs_accounted=True,
        )
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )

        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "RollingAverager",
            witness_names,
            source_file="rolling_averager.py",
            plan=plan,
        )

        assert "obj.buffer = state.buffer" in source
        assert "new_state = state" in source
        assert "return _ret_0, new_state" in source

    def test_canonical_planned_group_calls_only_selected_method(self):
        atom = MacroAtomSpec(
            name="Fit Update",
            description="Run only the update step",
            method_names=["fit_update"],
            inputs=[IOSpec(name="data", type_desc="np.ndarray", constraints="")],
            outputs=[IOSpec(name="result", type_desc="np.ndarray", constraints="")],
            concept_type=ConceptType.CUSTOM,
        )
        state_model = StateModelSpec(
            model_name="FitState",
            fields=[("prepared", "np.ndarray")],
        )
        operation = OperationSpec(
            operation_id="fit_pipeline",
            display_name="Fit Pipeline",
            role="state_transition",
            method_bindings=[
                MethodBinding(
                    method_name="fit_prepare",
                    signature=[ParameterFact(name="self"), ParameterFact(name="data")],
                ),
                MethodBinding(
                    method_name="fit_update",
                    signature=[ParameterFact(name="self"), ParameterFact(name="data")],
                ),
            ],
            direct_inputs=list(atom.inputs),
            required_state_slots=["prepared"],
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="return_value",
                    source_method="fit_update",
                )
            ],
            state_effects=[
                StateEffectSpec(
                    slot_name="prepared",
                    effect_kind="update",
                    source_method="fit_update",
                )
            ],
        )
        plan = ValidatedMacroPlan(
            plan=ProposedMacroPlan(
                macro_atoms=[atom],
                state_models=[state_model],
                canonical_ir=IngestIRPlan(
                    subject_name="Trainer",
                    operations=[operation],
                    state_slots=[StateSlotSpec(slot_name="prepared", state_kind="fitted", type_desc="np.ndarray")],
                ),
                planning_graph=IngestPlanGraph(
                    planned_groups=[
                        PlannedOperationGroup(
                            group_id="fit_pipeline__fit_update",
                            display_name="Fit Update",
                            group_role="state_transition",
                            member_operation_ids=["fit_pipeline"],
                            required_state_slots=["prepared"],
                            emitted_outputs=[
                                OutputBindingSpec(
                                    output_name="result",
                                    type_desc="np.ndarray",
                                    binding_kind="return_value",
                                    source_method="fit_update",
                                )
                            ],
                        )
                    ]
                ),
            ),
            all_attrs_accounted=True,
        )
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )

        source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            "Trainer",
            witness_names,
            source_file="trainer.py",
            plan=plan,
        )

        assert "obj.fit_update(data)" in source
        assert "obj.fit_prepare(data)" not in source


# ---------------------------------------------------------------------------
# Tests: Ghost witnesses with state
# ---------------------------------------------------------------------------


class TestStatefulGhostWitnesses:
    """Verify ghost witnesses gain state param when state_models exist."""

    def test_witness_has_state_param(self):
        plan = _make_stateful_plan()
        source, _ = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        assert "state: AbstractArray" in source

    def test_witness_returns_tuple(self):
        plan = _make_stateful_plan()
        source, _ = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        assert "tuple[" in source

    def test_witness_is_valid_python(self):
        plan = _make_stateful_plan()
        source, _ = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        ast.parse(source)

    def test_canonical_witness_uses_exact_inputs_and_marks_state_preserving(self):
        plan = _make_canonical_stateful_plan()
        source, _ = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
            plan=plan,
        )

        assert "def witness_average_computer(window_size: AbstractScalar, state: AbstractArray) -> tuple[AbstractScalar, AbstractArray]:" in source
        assert "state-updating" in source
        ast.parse(source)


# ---------------------------------------------------------------------------
# Tests: State edges
# ---------------------------------------------------------------------------


class TestStateEdges:
    """Verify deterministic state edge computation."""

    @pytest.fixture
    def ra_source(self, tmp_path) -> str:
        src = tmp_path / "rolling_averager.py"
        src.write_text(ROLLING_AVERAGER_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_state_edge_from_accumulator_to_computer(self, ra_source):
        """CDG should have state edge sample_accumulator -> average_computer."""
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        plan = _make_stateful_plan().plan
        edges = _compute_state_edges(dfg, plan.macro_atoms, plan.state_models)

        edge_pairs = {(e.source_id, e.target_id) for e in edges}
        assert ("sample_accumulator", "average_computer") in edge_pairs

    @pytest.mark.asyncio
    async def test_state_edge_typed_with_state_model(self, ra_source):
        """State edges should be typed with the state model name."""
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        plan = _make_stateful_plan().plan
        edges = _compute_state_edges(dfg, plan.macro_atoms, plan.state_models)

        for edge in edges:
            assert edge.source_type == "RollingAveragerState"
            assert edge.target_type == "RollingAveragerState"

    @pytest.mark.asyncio
    async def test_no_self_loop_edges(self, ra_source):
        """State edges should not include self-loops."""
        dfg = await extract_data_flow(ra_source, "RollingAverager")
        plan = _make_stateful_plan().plan
        edges = _compute_state_edges(dfg, plan.macro_atoms, plan.state_models)

        for edge in edges:
            assert edge.source_id != edge.target_id

    @pytest.mark.asyncio
    async def test_state_model_has_buffer_and_count(self, ra_source):
        """State model should include buffer and count fields."""
        plan = _make_stateful_plan()
        state_model = plan.plan.state_models[0]
        field_names = [f[0] for f in state_model.fields]
        assert "buffer" in field_names
        assert "count" in field_names


# ---------------------------------------------------------------------------
# Tests: Full stateful bundle (e2e)
# ---------------------------------------------------------------------------


class TestStatefulBundle:
    """End-to-end: agent.ingest() with a rolling-state class."""

    @pytest.fixture
    def ra_source(self, tmp_path) -> str:
        src = tmp_path / "rolling_averager.py"
        src.write_text(ROLLING_AVERAGER_SOURCE)
        return str(src)

    @pytest.mark.asyncio
    async def test_bundle_has_stateful_wrappers(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")

        assert "state: RollingAveragerState" in bundle.generated_atoms
        assert "RollingAverager.__new__" in bundle.generated_atoms
        assert "model_copy" in bundle.generated_atoms

    @pytest.mark.asyncio
    async def test_bundle_has_state_model_code(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")

        assert "RollingAveragerState" in bundle.generated_state_models
        assert "buffer" in bundle.generated_state_models
        assert "count" in bundle.generated_state_models
        ast.parse(bundle.generated_state_models)

    @pytest.mark.asyncio
    async def test_bundle_cdg_includes_state_edges(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")

        state_edges = [
            e for e in bundle.cdg.edges if e.source_type == "RollingAveragerState"
        ]
        assert len(state_edges) > 0

    @pytest.mark.asyncio
    async def test_bundle_match_results_verified(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")

        assert len(bundle.match_results) == 2
        for mr in bundle.match_results:
            assert mr.verified_match.verified is True

    @pytest.mark.asyncio
    async def test_bundle_generated_atoms_valid_python(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")
        ast.parse(bundle.generated_atoms)

    @pytest.mark.asyncio
    async def test_bundle_generated_witnesses_valid_python(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")
        ast.parse(bundle.generated_witnesses)

    @pytest.mark.asyncio
    async def test_bundle_witnesses_have_state(self, ra_source):
        agent, _ = _make_agent()
        bundle = await agent.ingest(ra_source, "RollingAverager")

        assert "state: AbstractArray" in bundle.generated_witnesses

    def test_bundle_materializes_runtime_state_from_canonical_plan(self):
        plan = _make_canonical_stateful_plan()
        plan = plan.model_copy(
            update={
                "plan": plan.plan.model_copy(
                    update={
                        "macro_atoms": [],
                        "state_models": [],
                    }
                ),
            }
        )

        bundle = emit_ingestion_bundle(plan, "RollingAverager")

        assert "class RollingAveragerState(BaseModel):" in bundle.generated_state_models
        assert "def average_computer(" in bundle.generated_atoms
        assert "state: RollingAveragerState" in bundle.generated_atoms
        assert bundle.cdg.metadata["canonical_semantics"] is True

"""Tests for Phase 3 code generation (sciona.ingester.emitter)."""

from __future__ import annotations

import ast


from sciona.architect.models import ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.ingester.emitter import (
    build_cdg_export,
    build_match_results,
    emit_ingestion_bundle,
    generate_atom_wrappers,
    generate_ghost_witnesses,
    generate_state_models,
)
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
# Helpers
# ---------------------------------------------------------------------------


def _make_plan() -> ValidatedMacroPlan:
    """Minimal validated plan with two macro-atoms and one state model."""
    atoms = [
        MacroAtomSpec(
            name="Signal Conditioner",
            description="Preprocess and filter signal",
            method_names=["__init__", "preprocess"],
            inputs=[
                IOSpec(name="raw", type_desc="np.ndarray", constraints="time domain")
            ],
            outputs=[
                IOSpec(
                    name="conditioned",
                    type_desc="np.ndarray",
                    constraints="time domain",
                )
            ],
            concept_type=ConceptType.SIGNAL_FILTER,
        ),
        MacroAtomSpec(
            name="Beat Detector",
            description="Detect beats in conditioned signal",
            method_names=["detect"],
            inputs=[
                IOSpec(name="signal", type_desc="np.ndarray", constraints="time domain")
            ],
            outputs=[
                IOSpec(name="onsets", type_desc="np.ndarray", constraints="int indices")
            ],
            concept_type=ConceptType.SIGNAL_TRANSFORM,
        ),
    ]
    edges = [
        DependencyEdge(
            source_id="signal_conditioner",
            target_id="beat_detector",
            output_name="conditioned",
            input_name="signal",
            source_type="np.ndarray",
            target_type="np.ndarray",
        ),
    ]
    state_models = [
        StateModelSpec(
            model_name="ProcessorState",
            fields=[("history", "list[float]"), ("threshold", "float")],
            source_attrs=["history", "threshold"],
            docstring="Cross-window state for the processor.",
        ),
    ]
    plan = ProposedMacroPlan(
        macro_atoms=atoms,
        edge_definitions=edges,
        state_models=state_models,
    )
    return ValidatedMacroPlan(
        plan=plan,
        all_attrs_accounted=True,
        coverage_report="All attributes accounted for.",
    )


def _make_canonical_plan(
    atom: MacroAtomSpec,
    *,
    operation: OperationSpec,
    state_models: list[StateModelSpec] | None = None,
    state_slots: list[StateSlotSpec] | None = None,
    planning_groups: list[PlannedOperationGroup] | None = None,
) -> ValidatedMacroPlan:
    return ValidatedMacroPlan(
        plan=ProposedMacroPlan(
            macro_atoms=[atom],
            state_models=state_models or [],
            canonical_ir=IngestIRPlan(
                subject_name="Estimator",
                operations=[operation],
                state_slots=state_slots or [],
            ),
            planning_graph=(
                IngestPlanGraph(planned_groups=planning_groups or [])
                if planning_groups
                else None
            ),
        ),
        all_attrs_accounted=True,
    )


# ---------------------------------------------------------------------------
# Tests: generate_state_models
# ---------------------------------------------------------------------------


class TestGenerateStateModels:
    def test_valid_python(self):
        specs = [
            StateModelSpec(
                model_name="BeatSQIState",
                fields=[("pool", "list[float]"), ("threshold", "float")],
                docstring="SQI state.",
            ),
        ]
        source = generate_state_models(specs)
        # Must be valid Python
        tree = ast.parse(source)
        # Must contain a class definition
        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
        assert len(classes) == 1
        assert classes[0].name == "BeatSQIState"

    def test_empty_specs(self):
        assert generate_state_models([]) == ""

    def test_sanitizes_free_form_type_descriptions(self):
        specs = [
            StateModelSpec(
                model_name="CalibrationState",
                fields=[
                    ("estimator", "Classifier or None"),
                    ("cv", "int | CV splitter | iterable"),
                    ("samples", "array-like"),
                    ("models", "list[CalibratedClassifier]"),
                    ("splits", "Iterable[tuple[array-like, array-like]]"),
                ],
            ),
        ]
        source = generate_state_models(specs)
        ast.parse(source)
        assert "estimator: object | None | None" in source
        assert "cv: int | object | Iterable[Any] | None" in source
        assert "samples: object | None" in source
        assert "models: list[object] | None" in source
        assert "splits: Iterable[tuple[object, object]] | None" in source

    def test_imports_numpy_when_stochastic_fields_exist(self):
        specs = [
            StateModelSpec(
                model_name="TraceState",
                stochastic={
                    "rng_field": "rng_key",
                    "rng_type": "jax.random.KeyArray",
                    "trace_field": "trace",
                },
            ),
        ]
        source = generate_state_models(specs)
        assert "import numpy as np" in source


# ---------------------------------------------------------------------------
# Tests: generate_atom_wrappers
# ---------------------------------------------------------------------------


class TestGenerateAtomWrappers:
    def test_register_atom_present(self):
        plan = _make_plan()
        _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)
        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
        )
        assert "@register_atom" in source
        assert "signal_conditioner" in source
        assert "beat_detector" in source

    def test_valid_python(self):
        plan = _make_plan()
        _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)
        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
        )
        # Must parse as valid Python
        ast.parse(source)

    def test_imports_generated_modules(self):
        plan = _make_plan()
        _, witness_names = generate_ghost_witnesses(
            plan.plan.macro_atoms,
            state_models=plan.plan.state_models,
        )
        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
        )
        assert "from state_models import ProcessorState" in source
        assert (
            "from witnesses import witness_beat_detector, witness_signal_conditioner"
            in source
        )
        assert "# mypy: disable-error-code=untyped-decorator" in source

    def test_canonical_wrapper_uses_exact_signature_and_return_value(self):
        atom = MacroAtomSpec(
            name="Predict",
            method_names=["predict"],
            inputs=[
                IOSpec(name="features", type_desc="np.ndarray", constraints=""),
                IOSpec(name="threshold", type_desc="float", constraints=""),
                IOSpec(name="unused", type_desc="float", constraints=""),
            ],
            outputs=[IOSpec(name="predictions", type_desc="np.ndarray", constraints="")],
            concept_type=ConceptType.SIGNAL_TRANSFORM,
        )
        operation = OperationSpec(
            operation_id="predict",
            display_name="Predict",
            role="predict",
            method_bindings=[
                MethodBinding(
                    method_name="predict",
                    signature=[
                        ParameterFact(name="self"),
                        ParameterFact(name="features"),
                        ParameterFact(name="threshold", kind="keyword_only"),
                    ],
                )
            ],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="predictions",
                    type_desc="np.ndarray",
                    binding_kind="return_value",
                    source_method="predict",
                )
            ],
        )
        plan = _make_canonical_plan(atom, operation=operation)
        _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)

        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
            class_name="Estimator",
            source_file="estimator.py",
            plan=plan,
        )

        assert "_ret_0 = obj.predict(features, threshold=threshold)" in source
        assert "unused" not in source.split("_ret_0 = obj.predict", 1)[1].split("\n", 1)[0]
        assert "return _ret_0" in source

    def test_canonical_wrapper_emits_tuple_and_attribute_bindings(self):
        atom = MacroAtomSpec(
            name="Summarize",
            method_names=["summarize"],
            inputs=[IOSpec(name="batch", type_desc="np.ndarray", constraints="")],
            outputs=[
                IOSpec(name="mean", type_desc="float", constraints=""),
                IOSpec(name="stdev", type_desc="float", constraints=""),
            ],
            concept_type=ConceptType.SIGNAL_TRANSFORM,
        )
        operation = OperationSpec(
            operation_id="summarize",
            display_name="Summarize",
            role="transform",
            method_bindings=[
                MethodBinding(
                    method_name="summarize",
                    signature=[ParameterFact(name="self"), ParameterFact(name="batch")],
                )
            ],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="mean",
                    type_desc="float",
                    binding_kind="tuple_element",
                    source_method="summarize",
                    tuple_index=0,
                ),
                OutputBindingSpec(
                    output_name="stdev",
                    type_desc="float",
                    binding_kind="attribute_read",
                    source_method="summarize",
                    source_attr="stdev_",
                ),
            ],
        )
        plan = _make_canonical_plan(atom, operation=operation)
        _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)

        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
            class_name="Estimator",
            source_file="estimator.py",
            plan=plan,
        )

        assert "return (_ret_0[0], obj.stdev_)" in source

    def test_canonical_wrapper_uses_planned_group_method_selection(self):
        atom = MacroAtomSpec(
            name="Fit Update",
            method_names=["fit_update"],
            inputs=[IOSpec(name="data", type_desc="np.ndarray", constraints="")],
            outputs=[IOSpec(name="result", type_desc="np.ndarray", constraints="")],
            concept_type=ConceptType.CUSTOM,
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
            emitted_outputs=[
                OutputBindingSpec(
                    output_name="result",
                    type_desc="np.ndarray",
                    binding_kind="return_value",
                    source_method="fit_update",
                )
            ],
        )
        plan = _make_canonical_plan(
            atom,
            operation=operation,
            planning_groups=[
                PlannedOperationGroup(
                    group_id="fit_pipeline__fit_update",
                    display_name="Fit Update",
                    group_role="state_transition",
                    member_operation_ids=["fit_pipeline"],
                    emitted_outputs=[
                        OutputBindingSpec(
                            output_name="result",
                            type_desc="np.ndarray",
                            binding_kind="return_value",
                            source_method="fit_update",
                        )
                    ],
                )
            ],
        )
        _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)

        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
            class_name="Estimator",
            source_file="estimator.py",
            plan=plan,
        )

        assert "obj.fit_update(data)" in source
        assert "obj.fit_prepare(data)" not in source

    def test_canonical_wrapper_fails_closed_when_outputs_are_underspecified(self):
        atom = MacroAtomSpec(
            name="Score",
            method_names=["score"],
            inputs=[IOSpec(name="features", type_desc="np.ndarray", constraints="")],
            outputs=[IOSpec(name="score", type_desc="float", constraints="")],
            concept_type=ConceptType.CUSTOM,
        )
        operation = OperationSpec(
            operation_id="score",
            display_name="Score",
            role="score",
            method_bindings=[
                MethodBinding(
                    method_name="score",
                    signature=[ParameterFact(name="self"), ParameterFact(name="features")],
                )
            ],
            direct_inputs=list(atom.inputs),
            emitted_outputs=[],
        )
        plan = _make_canonical_plan(atom, operation=operation)
        _, witness_names = generate_ghost_witnesses(plan.plan.macro_atoms)

        source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
            class_name="Estimator",
            source_file="estimator.py",
            plan=plan,
        )

        assert 'raise NotImplementedError("Score: canonical bindings resolved 0 outputs for 1 declared outputs")' in source
        assert "obj.score = " not in source


# ---------------------------------------------------------------------------
# Tests: generate_ghost_witnesses
# ---------------------------------------------------------------------------


class TestGenerateGhostWitnesses:
    def test_witness_per_atom(self):
        plan = _make_plan()
        source, name_map = generate_ghost_witnesses(plan.plan.macro_atoms)

        assert "witness_signal_conditioner" in source
        assert "witness_beat_detector" in source
        assert "Signal Conditioner" in name_map
        assert "Beat Detector" in name_map

    def test_valid_python(self):
        plan = _make_plan()
        source, _ = generate_ghost_witnesses(plan.plan.macro_atoms)
        ast.parse(source)


# ---------------------------------------------------------------------------
# Tests: build_cdg_export
# ---------------------------------------------------------------------------


class TestBuildCDGExport:
    def test_root_is_decomposed(self):
        plan = _make_plan()
        cdg = build_cdg_export(plan, "TestClass")

        root = next(n for n in cdg.nodes if "root" in n.node_id)
        assert root.status == NodeStatus.DECOMPOSED

    def test_children_are_atomic(self):
        plan = _make_plan()
        cdg = build_cdg_export(plan, "TestClass")

        children = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(children) == 2

    def test_typed_edges(self):
        plan = _make_plan()
        cdg = build_cdg_export(plan, "TestClass")

        assert len(cdg.edges) == 1
        edge = cdg.edges[0]
        assert edge.source_type == "np.ndarray"
        assert edge.target_type == "np.ndarray"

    def test_metadata(self):
        plan = _make_plan()
        cdg = build_cdg_export(plan, "TestClass")

        assert cdg.metadata["source"] == "ingester"
        assert cdg.metadata["class_name"] == "TestClass"

    def test_dedupes_duplicate_top_level_nodes(self):
        duplicate = MacroAtomSpec(
            name="Signal Conditioner",
            description="Duplicate node that should collapse into the first one",
            method_names=["other_preprocess"],
            inputs=[IOSpec(name="raw", type_desc="np.ndarray", constraints="time domain")],
            outputs=[IOSpec(name="conditioned", type_desc="np.ndarray", constraints="time domain")],
            concept_type=ConceptType.SIGNAL_FILTER,
        )
        plan = _make_plan()
        plan.plan.macro_atoms.append(duplicate)

        cdg = build_cdg_export(plan, "TestClass")

        node_ids = [node.node_id for node in cdg.nodes]
        assert node_ids.count("signal_conditioner") == 1
        root = next(node for node in cdg.nodes if node.node_id == "TestClass_root")
        assert root.children == ["signal_conditioner", "beat_detector"]


# ---------------------------------------------------------------------------
# Tests: build_match_results
# ---------------------------------------------------------------------------


class TestBuildMatchResults:
    def test_verified_true(self):
        plan = _make_plan()
        cdg = build_cdg_export(plan, "TestClass")
        results = build_match_results(cdg, "")

        assert len(results) == 2
        for mr in results:
            assert mr.success is True
            assert mr.verified_match is not None
            assert mr.verified_match.verified is True

    def test_verification_level(self):
        plan = _make_plan()
        cdg = build_cdg_export(plan, "TestClass")
        results = build_match_results(cdg, "")

        from sciona.types import VerificationLevel

        for mr in results:
            assert (
                mr.verified_match.verification_level == VerificationLevel.TYPE_CHECKED
            )


# ---------------------------------------------------------------------------
# Tests: emit_ingestion_bundle
# ---------------------------------------------------------------------------


class TestEmitIngestionBundle:
    def test_bundle_has_all_parts(self):
        plan = _make_plan()
        bundle = emit_ingestion_bundle(plan, "TestClass")

        assert bundle.generated_atoms != ""
        assert bundle.generated_witnesses != ""
        assert bundle.generated_state_models != ""
        assert len(bundle.cdg.nodes) > 0
        assert len(bundle.match_results) > 0

    def test_bundle_cdg_compatible_with_assembler(self):
        """CDG should have typed edges and atomic leaves."""
        plan = _make_plan()
        bundle = emit_ingestion_bundle(plan, "TestClass")

        atomic = [n for n in bundle.cdg.nodes if n.status == NodeStatus.ATOMIC]
        assert len(atomic) == 2
        for node in atomic:
            assert node.type_signature != ""

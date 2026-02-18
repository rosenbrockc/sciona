"""Tests for Phase 3 code generation (ageom.ingester.emitter)."""

from __future__ import annotations

import ast

import pytest

from ageom.architect.models import ConceptType, DependencyEdge, IOSpec, NodeStatus
from ageom.ingester.emitter import (
    build_cdg_export,
    build_match_results,
    emit_ingestion_bundle,
    generate_atom_wrappers,
    generate_ghost_witnesses,
    generate_state_models,
)
from ageom.ingester.models import (
    MacroAtomSpec,
    ProposedMacroPlan,
    StateModelSpec,
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
            inputs=[IOSpec(name="raw", type_desc="np.ndarray", constraints="time domain")],
            outputs=[IOSpec(name="conditioned", type_desc="np.ndarray", constraints="time domain")],
            concept_type=ConceptType.SIGNAL_FILTER,
        ),
        MacroAtomSpec(
            name="Beat Detector",
            description="Detect beats in conditioned signal",
            method_names=["detect"],
            inputs=[IOSpec(name="signal", type_desc="np.ndarray", constraints="time domain")],
            outputs=[IOSpec(name="onsets", type_desc="np.ndarray", constraints="int indices")],
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

        from ageom.types import VerificationLevel

        for mr in results:
            assert mr.verified_match.verification_level == VerificationLevel.TYPE_CHECKED


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

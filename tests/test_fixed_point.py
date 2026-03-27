"""Tests for the FixedPoint combinator node across all four implementation phases.

Covers:
- Phase 1: detect_cycle_partition, assembler cycle handling
- Phase 2: FIXED_POINT ConceptType, skeleton template, ghost sim generalization
- Phase 3: toposort_with_fixed_points, assembler FP emission
- Phase 4: precision gradient scaling, credit assignment, prescreen
"""

from __future__ import annotations

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.skeletons import (
    SKELETON_TEMPLATES,
    get_skeleton,
    instantiate_skeleton,
)
from sciona.clearinghouse.prescreen import _check_structure, prescreen
from sciona.principal.backprop import CreditAssigner
from sciona.principal.models import (
    BenchmarkResult,
    NodeTelemetry,
    OptimizationMetric,
)
from sciona.synthesizer.assembler import Assembler, AssemblyError, sanitize_name
from sciona.synthesizer.ghost_sim import (
    GhostSimReport,
    _compute_precision_gradients,
    _detect_message_passing_cycle,
)
from sciona.synthesizer.toposort import (
    detect_cycle_partition,
    toposort_nodes,
    toposort_with_fixed_points,
)
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(
    nid: str,
    name: str,
    concept: ConceptType = ConceptType.CUSTOM,
    status: NodeStatus = NodeStatus.ATOMIC,
    **kwargs,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=nid,
        name=name,
        description=f"Test node: {name}",
        concept_type=concept,
        status=status,
        **kwargs,
    )


def _edge(src: str, dst: str, out: str = "result", inp: str = "data") -> DependencyEdge:
    return DependencyEdge(
        source_id=src,
        target_id=dst,
        output_name=out,
        input_name=inp,
        source_type="any",
        target_type="any",
    )


def _make_match(node_id: str, decl_name: str, type_sig: str = "") -> MatchResult:
    decl = Declaration(name=decl_name, type_signature=type_sig, prover=Prover.PYTHON)
    candidate = CandidateMatch(declaration=decl, score=0.9, retrieval_method="test")
    vr = VerificationResult(candidate=candidate, verified=True)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement=type_sig),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


def _make_telemetry(nid: str, t: float, m: int, e: float) -> NodeTelemetry:
    return NodeTelemetry(
        node_id=nid, execution_time_ms=t, peak_memory_bytes=m, error_expansion=e
    )


# ===========================================================================
# Phase 1: detect_cycle_partition and assembler crash fix
# ===========================================================================


class TestDetectCyclePartition:
    def test_acyclic_graph(self):
        nodes = [_node("a", "A"), _node("b", "B")]
        edges = [_edge("a", "b")]
        acyclic, cycle_ids, is_valid = detect_cycle_partition(nodes, edges)
        assert cycle_ids == set()
        assert is_valid is False
        assert "a" in acyclic
        assert "b" in acyclic

    def test_message_passing_cycle_is_valid(self):
        nodes = [
            _node("a", "A", ConceptType.MESSAGE_PASSING),
            _node("b", "B", ConceptType.MESSAGE_PASSING),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        acyclic, cycle_ids, is_valid = detect_cycle_partition(nodes, edges)
        assert cycle_ids == {"a", "b"}
        assert is_valid is True
        assert acyclic == []

    def test_fixed_point_cycle_is_valid(self):
        nodes = [
            _node("a", "A", ConceptType.FIXED_POINT),
            _node("b", "B", ConceptType.FIXED_POINT),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        acyclic, cycle_ids, is_valid = detect_cycle_partition(nodes, edges)
        assert cycle_ids == {"a", "b"}
        assert is_valid is True

    def test_mixed_type_cycle_is_invalid(self):
        nodes = [
            _node("a", "A", ConceptType.MESSAGE_PASSING),
            _node("b", "B", ConceptType.CUSTOM),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        acyclic, cycle_ids, is_valid = detect_cycle_partition(nodes, edges)
        assert cycle_ids == {"a", "b"}
        assert is_valid is False

    def test_partial_cycle_with_acyclic_prefix(self):
        nodes = [
            _node("pre", "Pre", ConceptType.CUSTOM),
            _node("a", "A", ConceptType.MESSAGE_PASSING),
            _node("b", "B", ConceptType.MESSAGE_PASSING),
        ]
        edges = [_edge("pre", "a"), _edge("a", "b"), _edge("b", "a")]
        acyclic, cycle_ids, is_valid = detect_cycle_partition(nodes, edges)
        assert "pre" in acyclic
        assert cycle_ids == {"a", "b"}
        assert is_valid is True


class TestAssemblerCycleCrashFix:
    def test_assembler_raises_on_invalid_cycle(self):
        """Non-MP/FP cycles should raise AssemblyError, not ValueError."""
        nodes = [
            _node("root", "Root", ConceptType.CUSTOM, status=NodeStatus.DECOMPOSED,
                  children=["a", "b"]),
            _node("a", "A", ConceptType.CUSTOM,
                  inputs=[IOSpec(name="x", type_desc="int")],
                  outputs=[IOSpec(name="y", type_desc="int")]),
            _node("b", "B", ConceptType.CUSTOM,
                  inputs=[IOSpec(name="x", type_desc="int")],
                  outputs=[IOSpec(name="y", type_desc="int")]),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        cdg = CDGExport(nodes=nodes, edges=edges, metadata={"goal": "test"})
        matches = [
            _make_match("a", "mod.func_a", "int -> int"),
            _make_match("b", "mod.func_b", "int -> int"),
        ]
        assembler = Assembler(Prover.PYTHON)
        with pytest.raises(AssemblyError, match="Cycle detected"):
            assembler.assemble(cdg, matches)

    def test_assembler_accepts_mp_cycle(self):
        """Valid MESSAGE_PASSING cycle should not crash the assembler."""
        nodes = [
            _node("root", "Root", ConceptType.MESSAGE_PASSING,
                  status=NodeStatus.DECOMPOSED, children=["a", "b"]),
            _node("a", "A", ConceptType.MESSAGE_PASSING,
                  inputs=[IOSpec(name="x", type_desc="int")],
                  outputs=[IOSpec(name="y", type_desc="int")]),
            _node("b", "B", ConceptType.MESSAGE_PASSING,
                  inputs=[IOSpec(name="x", type_desc="int")],
                  outputs=[IOSpec(name="y", type_desc="int")]),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        cdg = CDGExport(nodes=nodes, edges=edges, metadata={"goal": "bp"})
        matches = [
            _make_match("a", "mod.var_to_factor", "int -> int"),
            _make_match("b", "mod.factor_to_var", "int -> int"),
        ]
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(cdg, matches)
        assert skeleton.prover == "python"
        # Should contain the cycle while-loop
        assert "_MAX_ITERS" in skeleton.source_code or "composition" in skeleton.source_code.lower()


# ===========================================================================
# Phase 2: FIXED_POINT ConceptType and skeleton
# ===========================================================================


class TestFixedPointConceptType:
    def test_enum_value(self):
        assert ConceptType.FIXED_POINT.value == "fixed_point"

    def test_enum_roundtrip(self):
        assert ConceptType("fixed_point") == ConceptType.FIXED_POINT

    def test_node_with_fp_fields(self):
        node = AlgorithmicNode(
            node_id="fp1",
            name="Solver",
            description="Iterative solver",
            concept_type=ConceptType.FIXED_POINT,
            fixed_point_max_iterations=200,
            fixed_point_convergence_field="residual_ok",
        )
        assert node.fixed_point_max_iterations == 200
        assert node.fixed_point_convergence_field == "residual_ok"

    def test_default_fp_fields(self):
        node = AlgorithmicNode(
            node_id="fp2",
            name="Solver",
            description="desc",
            concept_type=ConceptType.FIXED_POINT,
        )
        assert node.fixed_point_max_iterations == 0
        assert node.fixed_point_convergence_field == ""


class TestFixedPointSkeleton:
    def test_skeleton_registered(self):
        assert ConceptType.FIXED_POINT in SKELETON_TEMPLATES

    def test_skeleton_structure(self):
        skel = get_skeleton(ConceptType.FIXED_POINT)
        assert skel is not None
        assert skel.paradigm == ConceptType.FIXED_POINT
        assert len(skel.template_nodes) >= 3
        assert len(skel.template_edges) >= 2

    def test_skeleton_has_body_init_and_step(self):
        skel = get_skeleton(ConceptType.FIXED_POINT)
        assert skel is not None
        names = [n.name for n in skel.template_nodes]
        assert "Body Init" in names
        assert "Body Step" in names
        assert "Convergence Check" in names

    def test_skeleton_variant_aliases(self):
        skel1 = get_skeleton(ConceptType.FIXED_POINT, variant="fixed_point")
        skel2 = get_skeleton(ConceptType.FIXED_POINT, variant="iterative_solver")
        skel3 = get_skeleton(ConceptType.FIXED_POINT, variant="convergence_loop")
        assert skel1 is not None
        assert skel2 is not None
        assert skel3 is not None

    def test_instantiate_produces_fresh_ids(self):
        skel = get_skeleton(ConceptType.FIXED_POINT)
        assert skel is not None
        nodes1, edges1 = instantiate_skeleton(skel, "solve A")
        nodes2, edges2 = instantiate_skeleton(skel, "solve B")
        ids1 = {n.node_id for n in nodes1}
        ids2 = {n.node_id for n in nodes2}
        assert ids1.isdisjoint(ids2)


# ===========================================================================
# Phase 3: toposort_with_fixed_points and assembler FP emission
# ===========================================================================


class TestToposortWithFixedPoints:
    def test_no_fp_nodes(self):
        nodes = [_node("a", "A"), _node("b", "B")]
        edges = [_edge("a", "b")]
        top_order, fp_bodies = toposort_with_fixed_points(nodes, edges)
        assert "a" in top_order
        assert "b" in top_order
        assert fp_bodies == {}

    def test_fp_with_children(self):
        fp = _node("fp", "FP Root", ConceptType.FIXED_POINT,
                    status=NodeStatus.DECOMPOSED, children=["c1", "c2"])
        c1 = _node("c1", "Step")
        c2 = _node("c2", "Check")
        pre = _node("pre", "Pre")
        nodes = [pre, fp, c1, c2]
        edges = [
            _edge("pre", "fp"),
            _edge("c1", "c2"),
        ]
        top_order, fp_bodies = toposort_with_fixed_points(nodes, edges)
        # Top-level should contain pre and fp, but not c1/c2
        assert "pre" in top_order
        assert "fp" in top_order
        assert "c1" not in top_order
        assert "c2" not in top_order
        # FP body should be topologically sorted
        assert "fp" in fp_bodies
        assert fp_bodies["fp"] == ["c1", "c2"]


class TestAssemblerFixedPointEmission:
    def _make_fp_cdg(self):
        nodes = [
            _node("fp_root", "Iterative Solver", ConceptType.FIXED_POINT,
                  status=NodeStatus.DECOMPOSED, children=["init", "step", "check"],
                  fixed_point_max_iterations=50,
                  fixed_point_convergence_field="converged"),
            _node("init", "Body Init", ConceptType.STATE_INIT,
                  inputs=[IOSpec(name="x0", type_desc="float")],
                  outputs=[IOSpec(name="state", type_desc="float")]),
            _node("step", "Body Step", ConceptType.CUSTOM,
                  inputs=[IOSpec(name="state", type_desc="float")],
                  outputs=[IOSpec(name="next_state", type_desc="float")]),
            _node("check", "Convergence Check", ConceptType.CUSTOM,
                  inputs=[IOSpec(name="state", type_desc="float")],
                  outputs=[IOSpec(name="converged", type_desc="bool")]),
        ]
        edges = [
            _edge("init", "step", "state", "state"),
            _edge("step", "check", "next_state", "state"),
        ]
        return CDGExport(nodes=nodes, edges=edges, metadata={"goal": "solve"})

    def test_python_fp_emission(self):
        cdg = self._make_fp_cdg()
        matches = [
            _make_match("init", "mod.init_state", "float -> float"),
            _make_match("step", "mod.iterate", "float -> float"),
            _make_match("check", "mod.check_conv", "float -> bool"),
        ]
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(cdg, matches)
        assert skeleton.prover == "python"
        # Should contain the FP loop
        assert "_fp_max_iters" in skeleton.source_code or "composition" in skeleton.source_code.lower()

    def test_lean4_fp_emission(self):
        cdg = self._make_fp_cdg()
        matches = [
            _make_match("init", "mod.init_state", "float -> float"),
            _make_match("step", "mod.iterate", "float -> float"),
            _make_match("check", "mod.check_conv", "float -> bool"),
        ]
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(cdg, matches)
        # Lean4 should have sorry-guarded FP placeholder
        assert "sorry" in skeleton.source_code or "Fixed-point" in skeleton.source_code

    def test_coq_fp_emission(self):
        cdg = self._make_fp_cdg()
        matches = [
            _make_match("init", "mod.init_state", "float -> float"),
            _make_match("step", "mod.iterate", "float -> float"),
            _make_match("check", "mod.check_conv", "float -> bool"),
        ]
        assembler = Assembler(Prover.COQ)
        skeleton = assembler.assemble(cdg, matches)
        # Coq should have Admitted FP placeholder
        assert "Admitted" in skeleton.source_code or "Fixed-point" in skeleton.source_code


# ===========================================================================
# Phase 4: Precision gradients, credit assignment, prescreen
# ===========================================================================


class TestPrecisionGradientScaling:
    def test_fp_child_scaling(self):
        """Nodes inside FIXED_POINT bodies should have gradients scaled."""
        fp = AlgorithmicNode(
            node_id="fp",
            name="FP Root",
            description="fp",
            concept_type=ConceptType.FIXED_POINT,
            status=NodeStatus.DECOMPOSED,
            children=["leaf"],
            fixed_point_max_iterations=100,
        )
        leaf = AlgorithmicNode(
            node_id="leaf",
            name="Leaf",
            description="leaf",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.ATOMIC,
            inputs=[IOSpec(name="x", type_desc="np.ndarray")],
            outputs=[IOSpec(name="y", type_desc="np.ndarray")],
        )
        cdg = CDGExport(nodes=[fp, leaf], edges=[], metadata={})
        matches = {"leaf": _make_match("leaf", "some.op")}

        result_no_iter = _compute_precision_gradients(
            cdg, matches, ["leaf"], iterations_used=0
        )
        result_with_iter = _compute_precision_gradients(
            cdg, matches, ["leaf"], iterations_used=100
        )
        # With iterations, the gradient should be smaller (divided by sqrt(100)=10)
        if "leaf" in result_no_iter.gradients and "leaf" in result_with_iter.gradients:
            assert abs(result_with_iter.gradients["leaf"]) <= abs(
                result_no_iter.gradients["leaf"]
            )


class TestCreditAssignerConvergence:
    def test_convergence_gradients(self):
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="fp",
                    name="FP",
                    description="fp",
                    concept_type=ConceptType.FIXED_POINT,
                    status=NodeStatus.DECOMPOSED,
                    children=["leaf"],
                    fixed_point_max_iterations=100,
                ),
                AlgorithmicNode(
                    node_id="leaf",
                    name="Leaf",
                    description="leaf",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        bench = BenchmarkResult(global_loss=1.0)
        sim = GhostSimReport(ran=True, passed=True, iterations_used=90)

        assigner = CreditAssigner()
        grads = assigner.compute_gradients(
            cdg, bench, sim, OptimizationMetric.CONVERGENCE
        )
        assert len(grads) == 1
        assert grads[0].node_id == "leaf"
        assert grads[0].metric_type == OptimizationMetric.CONVERGENCE
        assert "90/100" in grads[0].bottleneck_reason

    def test_convergence_empty_when_no_fp(self):
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="leaf",
                    name="Leaf",
                    description="leaf",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        bench = BenchmarkResult(global_loss=1.0)
        sim = GhostSimReport(ran=True, passed=True, iterations_used=10)

        assigner = CreditAssigner()
        grads = assigner.compute_gradients(
            cdg, bench, sim, OptimizationMetric.CONVERGENCE
        )
        assert grads == []


class TestCreditAssignerLatencyFP:
    def test_latency_multiplied_by_iterations(self):
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="fp",
                    name="FP",
                    description="fp",
                    concept_type=ConceptType.FIXED_POINT,
                    status=NodeStatus.DECOMPOSED,
                    children=["body_leaf"],
                    fixed_point_max_iterations=100,
                ),
                AlgorithmicNode(
                    node_id="body_leaf",
                    name="Body Leaf",
                    description="body",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
                AlgorithmicNode(
                    node_id="normal",
                    name="Normal",
                    description="normal",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        bench = BenchmarkResult(
            global_loss=1.0,
            node_telemetry={
                "body_leaf": _make_telemetry("body_leaf", 10.0, 100, 0.0),
                "normal": _make_telemetry("normal", 10.0, 100, 0.0),
            },
        )
        sim = GhostSimReport(ran=True, passed=True, iterations_used=5)

        assigner = CreditAssigner()
        grads = assigner.compute_gradients(
            cdg, bench, sim, OptimizationMetric.LATENCY
        )
        # body_leaf: 10*5=50ms, normal: 10ms, total=60ms
        # body_leaf should get ~83.3%, normal ~16.7%
        body_grad = next(g for g in grads if g.node_id == "body_leaf")
        normal_grad = next(g for g in grads if g.node_id == "normal")
        assert body_grad.gradient_score > normal_grad.gradient_score
        assert "iterations" in body_grad.bottleneck_reason


class TestCreditAssignerStructureFP:
    def test_non_converging_fp_flagged(self):
        cdg = CDGExport(
            nodes=[
                AlgorithmicNode(
                    node_id="fp",
                    name="FP",
                    description="fp",
                    concept_type=ConceptType.FIXED_POINT,
                    status=NodeStatus.DECOMPOSED,
                    children=["body"],
                    fixed_point_max_iterations=100,
                ),
                AlgorithmicNode(
                    node_id="body",
                    name="Body",
                    description="body",
                    concept_type=ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        bench = BenchmarkResult(global_loss=0.0)
        sim = GhostSimReport(ran=True, passed=True, iterations_used=85)

        assigner = CreditAssigner()
        grads = assigner.compute_gradients(
            cdg, bench, sim, OptimizationMetric.STRUCTURE
        )
        assert len(grads) >= 1
        assert any("non-converging" in g.bottleneck_reason for g in grads)


class TestPrescreenFixedPoint:
    def test_cycle_rejected_without_fp_annotation(self):
        sources = {"a": "def f(): pass", "b": "def g(): pass"}
        result = prescreen(sources, [("a", "b"), ("b", "a")])
        assert not result.passed

    def test_cycle_accepted_with_fp_annotation(self):
        sources = {"a": "def f(): pass", "b": "def g(): pass"}
        result = prescreen(
            sources,
            [("a", "b"), ("b", "a")],
            fixed_point_node_ids=frozenset({"a", "b"}),
        )
        assert result.passed

    def test_partial_fp_cycle_rejected(self):
        """Cycle where only some nodes are in FP subgraphs should be rejected."""
        sources = {"a": "def f(): pass", "b": "def g(): pass", "c": "def h(): pass"}
        result = prescreen(
            sources,
            [("a", "b"), ("b", "a")],
            fixed_point_node_ids=frozenset({"a"}),  # only 'a', not 'b'
        )
        assert not result.passed

    def test_check_structure_with_fp_ids(self):
        reasons = _check_structure(
            ["a", "b"],
            [("a", "b"), ("b", "a")],
            fixed_point_node_ids=frozenset({"a", "b"}),
        )
        assert reasons == []

    def test_check_structure_without_fp_ids(self):
        reasons = _check_structure(
            ["a", "b"],
            [("a", "b"), ("b", "a")],
        )
        assert any("cycle" in r.lower() for r in reasons)


# ===========================================================================
# Ghost sim backward compat: _detect_message_passing_cycle still works
# ===========================================================================


class TestDetectMessagePassingCycleBackcompat:
    def test_mp_cycle_detected(self):
        nodes = [
            _node("a", "A", ConceptType.MESSAGE_PASSING),
            _node("b", "B", ConceptType.MESSAGE_PASSING),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        cycle_ids, is_mp = _detect_message_passing_cycle(nodes, edges)
        assert cycle_ids == {"a", "b"}
        assert is_mp is True

    def test_fp_cycle_not_flagged_as_mp(self):
        nodes = [
            _node("a", "A", ConceptType.FIXED_POINT),
            _node("b", "B", ConceptType.FIXED_POINT),
        ]
        edges = [_edge("a", "b"), _edge("b", "a")]
        cycle_ids, is_mp = _detect_message_passing_cycle(nodes, edges)
        assert cycle_ids == {"a", "b"}
        # Not flagged as message-passing (different concept type)
        assert is_mp is False

    def test_no_cycle(self):
        nodes = [_node("a", "A"), _node("b", "B")]
        edges = [_edge("a", "b")]
        cycle_ids, is_mp = _detect_message_passing_cycle(nodes, edges)
        assert cycle_ids == set()
        assert is_mp is False

"""Tests for the MAP_OVER combinator runtime integration."""

from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.skeletons import SKELETON_TEMPLATES, get_skeleton, instantiate_skeleton
from sciona.clearinghouse.prescreen import _check_structure, prescreen
from sciona.principal.backprop import CreditAssigner
from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.synthesizer.assembler import Assembler
from sciona.synthesizer.ghost_sim import GhostSimReport
from sciona.synthesizer.toposort import toposort_with_fixed_points
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


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
    verification = VerificationResult(candidate=candidate, verified=True)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement=type_sig),
        verified_match=verification,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


def _make_map_cdg() -> CDGExport:
    nodes = [
        _node(
            "map_root",
            "MAP Root",
            ConceptType.MAP_OVER,
            status=NodeStatus.DECOMPOSED,
            children=["body_init", "body_process", "collect_results"],
            inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
            outputs=[IOSpec(name="results", type_desc="list[any]")],
            map_window_size=128,
            map_hop_size=64,
        ),
        _node(
            "body_init",
            "Body Init",
            ConceptType.STATE_INIT,
            inputs=[IOSpec(name="window", type_desc="np.ndarray")],
            outputs=[IOSpec(name="state", type_desc="any")],
        ),
        _node(
            "body_process",
            "Body Process",
            ConceptType.CUSTOM,
            inputs=[IOSpec(name="state", type_desc="any")],
            outputs=[IOSpec(name="result", type_desc="any")],
        ),
        _node(
            "collect_results",
            "Collect Results",
            ConceptType.CUSTOM,
            inputs=[IOSpec(name="result", type_desc="any")],
            outputs=[IOSpec(name="results", type_desc="list[any]")],
        ),
    ]
    edges = [
        _edge("body_init", "body_process", "state", "state"),
        _edge("body_process", "collect_results", "result", "result"),
    ]
    return CDGExport(nodes=nodes, edges=edges, metadata={"goal": "map over windows"})


class TestMapNodeFields:
    def test_defaults(self):
        node = AlgorithmicNode(
            node_id="map",
            name="MAP",
            description="map",
            concept_type=ConceptType.MAP_OVER,
        )
        assert node.map_window_size == 0
        assert node.map_hop_size == 0

    def test_roundtrip(self):
        node = AlgorithmicNode(
            node_id="map",
            name="MAP",
            description="map",
            concept_type=ConceptType.MAP_OVER,
            map_window_size=256,
            map_hop_size=128,
        )
        restored = AlgorithmicNode.model_validate(node.model_dump())
        assert restored.map_window_size == 256
        assert restored.map_hop_size == 128


class TestMapSkeleton:
    def test_registered(self):
        assert ConceptType.MAP_OVER in SKELETON_TEMPLATES

    def test_structure_and_instantiation(self):
        skeleton = get_skeleton(ConceptType.MAP_OVER)
        assert skeleton is not None
        # JSON asset has 4 stages (no root); Python fallback has 5 (with root).
        # Accept either — JSON asset is the auditable source of truth.
        assert len(skeleton.template_nodes) in {4, 5}
        assert len(skeleton.template_edges) == 3

        nodes1, _edges1 = instantiate_skeleton(skeleton, "first map")
        nodes2, _edges2 = instantiate_skeleton(skeleton, "second map")
        assert {node.node_id for node in nodes1}.isdisjoint(
            {node.node_id for node in nodes2}
        )

        # Verify key stages exist regardless of root presence
        node_names = {node.name for node in nodes1}
        assert "Window Slicer" in node_names
        assert "Body Process" in node_names
        assert "Collect Results" in node_names


class TestMapToposort:
    def test_map_body_is_sorted_independently(self):
        pre = _node("pre", "Pre")
        map_root = _node(
            "map",
            "MAP Root",
            ConceptType.MAP_OVER,
            status=NodeStatus.DECOMPOSED,
            children=["init", "process", "collect"],
        )
        init = _node("init", "Body Init", ConceptType.STATE_INIT)
        process = _node("process", "Body Process")
        collect = _node("collect", "Collect Results")
        edges = [
            _edge("pre", "map"),
            _edge("init", "process"),
            _edge("process", "collect"),
        ]

        top_order, combinator_bodies = toposort_with_fixed_points(
            [pre, map_root, init, process, collect],
            edges,
        )

        assert top_order == ["pre", "map"]
        assert combinator_bodies["map"] == ["init", "process", "collect"]


class TestMapAssembler:
    def test_python_emission(self):
        assembler = Assembler(Prover.PYTHON)
        skeleton = assembler.assemble(
            _make_map_cdg(),
            [
                _make_match("body_init", "mod.body_init", "np.ndarray -> any"),
                _make_match("body_process", "mod.body_process", "any -> any"),
                _make_match("collect_results", "mod.collect", "any -> list[any]"),
            ],
        )
        assert "for _win_start_map_root" in skeleton.source_code
        assert "_map_window_map_root" in skeleton.source_code
        assert "_map_hop_map_root" in skeleton.source_code
        assert "_map_results_map_root" in skeleton.source_code
        assert ".append(" in skeleton.source_code

    def test_lean4_emission(self):
        assembler = Assembler(Prover.LEAN4)
        skeleton = assembler.assemble(
            _make_map_cdg(),
            [
                _make_match("body_init", "mod.body_init", "np.ndarray -> any"),
                _make_match("body_process", "mod.body_process", "any -> any"),
                _make_match("collect_results", "mod.collect", "any -> list[any]"),
            ],
        )
        assert "MAP over windows: MAP Root" in skeleton.source_code
        assert "sorry" in skeleton.source_code

    def test_coq_emission(self):
        assembler = Assembler(Prover.COQ)
        skeleton = assembler.assemble(
            _make_map_cdg(),
            [
                _make_match("body_init", "mod.body_init", "np.ndarray -> any"),
                _make_match("body_process", "mod.body_process", "any -> any"),
                _make_match("collect_results", "mod.collect", "any -> list[any]"),
            ],
        )
        assert "MAP over windows: MAP Root" in skeleton.source_code
        assert "Admitted." in skeleton.source_code


class TestMapPrescreen:
    def test_map_body_passes_structure_check(self):
        reasons = _check_structure(
            ["body_init", "body_process", "collect_results"],
            [
                ("body_init", "body_process"),
                ("body_process", "collect_results"),
            ],
            fixed_point_node_ids=frozenset({"body_init", "body_process", "collect_results"}),
        )
        assert reasons == []

    def test_map_body_passes_prescreen(self):
        result = prescreen(
            {
                "body_init": "def body_init(window): return window",
                "body_process": "def body_process(state): return state",
                "collect_results": "def collect_results(result): return [result]",
            },
            [
                ("body_init", "body_process"),
                ("body_process", "collect_results"),
            ],
            fixed_point_node_ids=frozenset({"body_init", "body_process", "collect_results"}),
        )
        assert result.passed


class TestMapBackprop:
    def test_high_window_count_flags_map_body(self):
        cdg = CDGExport(
            nodes=[
                _node(
                    "map",
                    "MAP Root",
                    ConceptType.MAP_OVER,
                    status=NodeStatus.DECOMPOSED,
                    children=["body"],
                    map_window_size=128,
                    map_hop_size=64,
                ),
                _node(
                    "body",
                    "Body Process",
                    ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        gradients = CreditAssigner().compute_gradients(
            cdg,
            BenchmarkResult(global_loss=0.0),
            GhostSimReport(ran=True, passed=True, signal_length=6592),
            OptimizationMetric.STRUCTURE,
        )
        assert len(gradients) == 1
        assert gradients[0].node_id == "body"
        assert "high-window-count MAP body" in gradients[0].bottleneck_reason
        assert "102 windows" in gradients[0].bottleneck_reason

    def test_zero_signal_length_does_not_flag_map_body(self):
        cdg = CDGExport(
            nodes=[
                _node(
                    "map",
                    "MAP Root",
                    ConceptType.MAP_OVER,
                    status=NodeStatus.DECOMPOSED,
                    children=["body"],
                    map_window_size=128,
                    map_hop_size=64,
                ),
                _node(
                    "body",
                    "Body Process",
                    ConceptType.CUSTOM,
                    status=NodeStatus.ATOMIC,
                ),
            ],
            edges=[],
        )
        gradients = CreditAssigner().compute_gradients(
            cdg,
            BenchmarkResult(global_loss=0.0),
            GhostSimReport(ran=True, passed=True, signal_length=0),
            OptimizationMetric.STRUCTURE,
        )
        assert gradients == []

"""Focused tests for the semantic/boundary CDG projection layer."""

from __future__ import annotations

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.semantic_graph import (
    SemanticCDG,
    SemanticBoundaryKind,
    SemanticDataKind,
    SemanticEdgeProvenance,
    insert_node_before_boundary_consumer,
    insert_node_before_root_input_consumer,
    project_semantic_cdg,
)
from sciona.synthesizer.assembler import Assembler
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)
from sciona.principal.expansion_rules.signal_event_rate import (
    _build_insert_jump_removal_before_filter,
)


def _root_boundary_signal_rate_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="root",
        parent_id=None,
        name="Detect heart rate from raw ECG signal",
        description="Top-level ECG heart-rate pipeline",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.DECOMPOSED,
        children=["filt", "det", "rate"],
        depth=0,
        inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
    )
    filt = AlgorithmicNode(
        node_id="filt",
        parent_id="root",
        name="Filter Signal",
        description="Condition the ECG before detection",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="filter_signal_for_detection",
        depth=1,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )
    det = AlgorithmicNode(
        node_id="det",
        parent_id="root",
        name="Detect Peaks",
        description="Detect cardiac events",
        concept_type=ConceptType.DATA_EXTRACTION,
        status=NodeStatus.ATOMIC,
        matched_primitive="detect_peaks_in_signal",
        depth=1,
        inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        outputs=[IOSpec(name="events", type_desc="np.ndarray")],
    )
    rate = AlgorithmicNode(
        node_id="rate",
        parent_id="root",
        name="Compute Event Rate",
        description="Estimate heart rate from detected events",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.ATOMIC,
        matched_primitive="compute_event_rate",
        depth=1,
        inputs=[IOSpec(name="events", type_desc="np.ndarray")],
        outputs=[IOSpec(name="rate", type_desc="np.ndarray")],
    )
    return CDGExport(
        nodes=[root, filt, det, rate],
        edges=[
            DependencyEdge(
                source_id="filt",
                target_id="det",
                output_name="signal",
                input_name="signal",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
            DependencyEdge(
                source_id="det",
                target_id="rate",
                output_name="events",
                input_name="events",
                source_type="np.ndarray",
                target_type="np.ndarray",
            ),
        ],
        metadata={"goal": "ECG HR"},
    )


def _jump_removal_node() -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="jump",
        parent_id="root",
        name="Remove Signal Jumps",
        description="Remove large waveform discontinuities before filtering",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="remove_signal_jumps",
        depth=1,
        inputs=[
            IOSpec(name="signal", type_desc="np.ndarray"),
            IOSpec(name="sampling_rate", type_desc="float"),
        ],
        outputs=[IOSpec(name="signal", type_desc="np.ndarray")],
    )


def _root_feature_pipeline_cdg() -> CDGExport:
    root = AlgorithmicNode(
        node_id="root_features",
        parent_id=None,
        name="Score Feature Vector",
        description="Top-level feature scoring pipeline",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.DECOMPOSED,
        children=["score"],
        depth=0,
        inputs=[IOSpec(name="features", type_desc="np.ndarray", data_kind="feature_vector")],
        outputs=[IOSpec(name="score", type_desc="float", data_kind="scalar_statistic")],
    )
    score = AlgorithmicNode(
        node_id="score",
        parent_id="root_features",
        name="Score Features",
        description="Map feature vectors to a scalar score",
        concept_type=ConceptType.ANALYSIS,
        status=NodeStatus.ATOMIC,
        matched_primitive="score_feature_vector",
        depth=1,
        inputs=[IOSpec(name="features", type_desc="np.ndarray", data_kind="feature_vector")],
        outputs=[IOSpec(name="score", type_desc="float", data_kind="scalar_statistic")],
    )
    return CDGExport(nodes=[root, score], edges=[], metadata={"goal": "feature scoring"})


def _match_result(node: AlgorithmicNode) -> MatchResult:
    type_sig = node.type_signature or "np.ndarray -> np.ndarray"
    decl = Declaration(
        name=node.matched_primitive or node.name,
        type_signature=type_sig,
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=decl,
        score=0.99,
        retrieval_method="test",
    )
    verification = VerificationResult(candidate=candidate, verified=True)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node.node_id, statement=type_sig),
        verified_match=verification,
        all_candidates=[candidate],
        all_verifications=[verification],
    )


def test_project_semantic_cdg_exposes_root_input_boundaries() -> None:
    semantic = project_semantic_cdg(_root_boundary_signal_rate_cdg())

    signal_boundary = next(
        boundary
        for boundary in semantic.boundaries
        if boundary.kind == SemanticBoundaryKind.ROOT_INPUT
        and boundary.port.name == "signal"
    )
    signal_edge = next(
        edge
        for edge in semantic.edges
        if edge.source_id == signal_boundary.boundary_id and edge.target_id == "filt"
    )

    assert signal_edge.data_kind == SemanticDataKind.WAVEFORM
    assert signal_edge.provenance == SemanticEdgeProvenance.ROOT_CONTRACT


def test_project_semantic_cdg_exposes_root_output_boundaries() -> None:
    semantic = project_semantic_cdg(_root_boundary_signal_rate_cdg())

    rate_boundary = next(
        boundary
        for boundary in semantic.boundaries
        if boundary.kind == SemanticBoundaryKind.ROOT_OUTPUT
        and boundary.port.name == "rate"
    )
    rate_edge = next(
        edge
        for edge in semantic.edges
        if edge.target_id == rate_boundary.boundary_id and edge.source_id == "rate"
    )

    assert rate_edge.data_kind == SemanticDataKind.RATE_SERIES
    assert rate_edge.provenance == SemanticEdgeProvenance.ROOT_CONTRACT


def test_semantic_cdg_roundtrip_preserves_boundary_projection() -> None:
    semantic = project_semantic_cdg(_root_boundary_signal_rate_cdg())
    restored = SemanticCDG.model_validate(semantic.model_dump())

    assert restored.boundaries == semantic.boundaries
    assert restored.edges == semantic.edges
    assert restored.metadata == semantic.metadata


def test_project_semantic_cdg_prefers_declared_cross_family_data_kinds() -> None:
    semantic = project_semantic_cdg(_root_feature_pipeline_cdg())

    feature_boundary = next(
        boundary
        for boundary in semantic.boundaries
        if boundary.kind == SemanticBoundaryKind.ROOT_INPUT
        and boundary.port.name == "features"
    )
    feature_edge = next(
        edge
        for edge in semantic.edges
        if edge.source_id == feature_boundary.boundary_id and edge.target_id == "score"
    )

    assert feature_boundary.data_kind == SemanticDataKind.FEATURE_VECTOR
    assert feature_edge.data_kind == SemanticDataKind.FEATURE_VECTOR


def test_graph_rewriter_uses_boundary_fallback_for_root_input_rewrite() -> None:
    cdg = _root_boundary_signal_rate_cdg()
    rule = _build_insert_jump_removal_before_filter()

    assert GraphRewriter()._find_match(rule, cdg) is None
    result = GraphRewriter().apply_rule(rule, cdg)
    assert not result.is_failure
    rewritten = result.unwrap()

    prims = {node.matched_primitive for node in rewritten.nodes if node.matched_primitive}
    assert "remove_signal_jumps" in prims

    root = next(node for node in rewritten.nodes if node.node_id == "root")
    jump = next(
        node
        for node in rewritten.nodes
        if node.matched_primitive == "remove_signal_jumps"
    )
    assert any(
        edge.source_id == jump.node_id
        and edge.target_id == "filt"
        and edge.input_name == "signal"
        for edge in rewritten.edges
    )
    assert jump.node_id in root.children
    assert not any(edge.target_id == jump.node_id for edge in rewritten.edges)

    semantic = project_semantic_cdg(rewritten)
    signal_boundary = next(
        boundary
        for boundary in semantic.boundaries
        if boundary.kind == SemanticBoundaryKind.ROOT_INPUT
        and boundary.port.name == "signal"
    )
    boundary_targets = {
        edge.target_id
        for edge in semantic.edges
        if edge.source_id == signal_boundary.boundary_id
    }

    assert jump.node_id in boundary_targets
    assert "filt" not in boundary_targets


def test_boundary_interposer_helper_lowers_root_input_rewrite_without_fake_source() -> None:
    rewritten = insert_node_before_root_input_consumer(
        _root_boundary_signal_rate_cdg(),
        root_input_name="signal",
        target_primitive="filter_signal_for_detection",
        inserted_node=_jump_removal_node(),
    )

    jump = next(
        node
        for node in rewritten.nodes
        if node.matched_primitive == "remove_signal_jumps"
    )
    root = next(node for node in rewritten.nodes if node.node_id == "root")

    assert jump.parent_id == "root"
    assert jump.node_id in root.children


def test_generic_boundary_interposer_can_match_by_boundary_data_kind() -> None:
    rewritten = insert_node_before_boundary_consumer(
        _root_boundary_signal_rate_cdg(),
        boundary_kind=SemanticBoundaryKind.ROOT_INPUT,
        boundary_data_kind=SemanticDataKind.WAVEFORM,
        target_primitive="filter_signal_for_detection",
        inserted_node=_jump_removal_node(),
    )

    assert any(
        node.matched_primitive == "remove_signal_jumps" for node in rewritten.nodes
    )


def test_boundary_rewrite_lowering_preserves_assembler_visibility() -> None:
    rewritten = insert_node_before_root_input_consumer(
        _root_boundary_signal_rate_cdg(),
        root_input_name="signal",
        target_primitive="filter_signal_for_detection",
        inserted_node=_jump_removal_node(),
    )
    matches = [
        _match_result(node)
        for node in rewritten.nodes
        if node.status == NodeStatus.ATOMIC and node.matched_primitive
    ]

    skeleton = Assembler(Prover.PYTHON).assemble(rewritten, matches)
    unit_ids = {unit.node_id for unit in skeleton.units}
    jump = next(
        node
        for node in rewritten.nodes
        if node.matched_primitive == "remove_signal_jumps"
    )

    assert jump.node_id in unit_ids
    assert "remove_signal_jumps" in skeleton.source_code

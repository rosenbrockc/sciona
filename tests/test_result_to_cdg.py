"""Tests for the OrchestratorResult-to-CDG converter."""

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.orchestrator import OrchestratorResult
from sciona.result_to_cdg import RunCDGMetadata, orchestrator_result_to_cdg
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    VerificationResult,
)


def _make_root_node() -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="root",
        parent_id=None,
        name="Root Goal",
        description="Top-level goal",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.DECOMPOSED,
        children=["atom_a", "atom_b"],
        depth=0,
    )


def _make_atomic_node(node_id: str, name: str) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        parent_id="root",
        name=name,
        description=f"Test {name}",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        type_signature="nat -> nat",
        inputs=[IOSpec(name="x", type_desc="nat")],
        outputs=[IOSpec(name="y", type_desc="nat")],
        depth=1,
    )


def _make_match_result(node_id: str, success: bool, decl_name: str = "") -> MatchResult:
    decl_name = decl_name or f"decl_{node_id}"
    decl = Declaration(name=decl_name, type_signature="nat -> nat")
    candidate = CandidateMatch(
        declaration=decl, score=0.9, retrieval_method="embedding"
    )
    vr = VerificationResult(candidate=candidate, verified=success)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement="nat -> nat"),
        verified_match=vr if success else None,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


def _make_metadata() -> RunCDGMetadata:
    return RunCDGMetadata(
        run_id="run-001",
        goal="Sort a list",
        execution_path="verified",
        timestamp="2026-03-17T00:00:00Z",
        verified_leaf_coverage=0.0,
    )


def _make_cdg_and_result(
    match_a: bool = True, match_b: bool = False
) -> OrchestratorResult:
    root = _make_root_node()
    atom_a = _make_atomic_node("atom_a", "Atom A")
    atom_b = _make_atomic_node("atom_b", "Atom B")
    cdg = CDGExport(nodes=[root, atom_a, atom_b], edges=[])
    match_results = [
        _make_match_result("atom_a", match_a),
        _make_match_result("atom_b", match_b),
    ]
    return OrchestratorResult(
        cdg=cdg,
        match_results=match_results,
        rounds_used=1,
    )


def test_basic_conversion():
    """Create a simple OrchestratorResult with 2 atomic nodes, 1 matched, and verify structure."""
    result = _make_cdg_and_result(match_a=True, match_b=False)
    metadata = _make_metadata()

    cdg_dict = orchestrator_result_to_cdg(result, metadata)

    assert "nodes" in cdg_dict
    assert "edges" in cdg_dict
    assert len(cdg_dict["nodes"]) == 3
    # Verify node_ids are present
    node_ids = {n["node_id"] for n in cdg_dict["nodes"]}
    assert node_ids == {"root", "atom_a", "atom_b"}


def test_verified_leaf_coverage_computation():
    """Verify coverage fraction is correct: 1 of 2 atomic nodes matched = 0.5."""
    result = _make_cdg_and_result(match_a=True, match_b=False)
    metadata = _make_metadata()

    orchestrator_result_to_cdg(result, metadata)

    assert metadata.verified_leaf_coverage == 0.5

    # Now test with both matched
    result_all = _make_cdg_and_result(match_a=True, match_b=True)
    metadata_all = _make_metadata()
    orchestrator_result_to_cdg(result_all, metadata_all)
    assert metadata_all.verified_leaf_coverage == 1.0

    # None matched
    result_none = _make_cdg_and_result(match_a=False, match_b=False)
    metadata_none = _make_metadata()
    orchestrator_result_to_cdg(result_none, metadata_none)
    assert metadata_none.verified_leaf_coverage == 0.0


def test_matched_primitive_enrichment():
    """Verify matched_primitive is set on the correct node."""
    result = _make_cdg_and_result(match_a=True, match_b=False)
    metadata = _make_metadata()

    cdg_dict = orchestrator_result_to_cdg(result, metadata)

    nodes_by_id = {n["node_id"]: n for n in cdg_dict["nodes"]}
    # atom_a was matched => matched_primitive should be set
    assert nodes_by_id["atom_a"]["matched_primitive"] == "decl_atom_a"
    # atom_b was not matched => matched_primitive stays None
    assert nodes_by_id["atom_b"]["matched_primitive"] is None


def test_provenance_on_root_node():
    """Verify run_id, goal, timestamp, execution_path appear on the root node."""
    result = _make_cdg_and_result(match_a=True, match_b=False)
    metadata = _make_metadata()

    cdg_dict = orchestrator_result_to_cdg(result, metadata)

    root = next(n for n in cdg_dict["nodes"] if n["node_id"] == "root")
    assert "provenance" in root
    prov = root["provenance"]
    assert prov["run_id"] == "run-001"
    assert prov["goal"] == "Sort a list"
    assert prov["timestamp"] == "2026-03-17T00:00:00Z"
    assert prov["execution_path"] == "verified"
    assert prov["verified_leaf_coverage"] == 0.5

    # Non-root nodes should NOT have provenance
    atom_a = next(n for n in cdg_dict["nodes"] if n["node_id"] == "atom_a")
    assert "provenance" not in atom_a

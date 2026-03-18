"""End-to-end tests for CDG export/import roundtrip, result_to_cdg conversion,
and exemplar JSON validity.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ageom.architect.handoff import CDGExport, save_json, load_json
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.orchestrator import OrchestratorResult
from ageom.result_to_cdg import RunCDGMetadata, orchestrator_result_to_cdg
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    VerificationResult,
)
from ageom.upsert_cdg import sanitize_cdg

# ---------------------------------------------------------------------------
# Helpers (reuse patterns from test_result_to_cdg)
# ---------------------------------------------------------------------------

EXEMPLAR_DIR = Path(__file__).resolve().parent.parent / "ageom" / "data" / "exemplars"


def _make_atomic_node(node_id: str, name: str, parent_id: str = "root") -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        parent_id=parent_id,
        name=name,
        description=f"Test {name}",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        type_signature="nat -> nat",
        inputs=[IOSpec(name="x", type_desc="nat")],
        outputs=[IOSpec(name="y", type_desc="nat")],
        depth=1,
    )


def _make_root_node(children: list[str] | None = None) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="root",
        parent_id=None,
        name="Root Goal",
        description="Top-level goal",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.DECOMPOSED,
        children=children or [],
        depth=0,
    )


def _make_edge(source_id: str, target_id: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name="out",
        input_name="inp",
        source_type="nat",
        target_type="nat",
    )


def _make_match_result(node_id: str, success: bool) -> MatchResult:
    decl = Declaration(name=f"decl_{node_id}", type_signature="nat -> nat")
    candidate = CandidateMatch(declaration=decl, score=0.9, retrieval_method="embedding")
    vr = VerificationResult(candidate=candidate, verified=success)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement="nat -> nat"),
        verified_match=vr if success else None,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


def _make_metadata() -> RunCDGMetadata:
    return RunCDGMetadata(
        run_id="run-e2e-001",
        goal="Test goal",
        execution_path="verified",
        timestamp="2026-03-17T00:00:00Z",
        verified_leaf_coverage=0.0,
    )


# ---------------------------------------------------------------------------
# Test 1: CDG save/load roundtrip
# ---------------------------------------------------------------------------


def test_cdg_save_load_roundtrip(tmp_path: Path) -> None:
    """Build a CDG, save_json(), load_json(), assert nodes/edges/metadata preserved."""
    root = _make_root_node(children=["a1", "a2"])
    a1 = _make_atomic_node("a1", "Atom A")
    a2 = _make_atomic_node("a2", "Atom B")
    edge = _make_edge("a1", "a2")

    cdg = CDGExport(
        nodes=[root, a1, a2],
        edges=[edge],
        metadata={"goal": "roundtrip test", "extra": 42},
    )

    path = tmp_path / "test_cdg.json"
    save_json(cdg, path)
    loaded = load_json(path)

    # Nodes preserved
    assert len(loaded.nodes) == 3
    loaded_ids = {n.node_id for n in loaded.nodes}
    assert loaded_ids == {"root", "a1", "a2"}

    # Edges preserved
    assert len(loaded.edges) == 1
    assert loaded.edges[0].source_id == "a1"
    assert loaded.edges[0].target_id == "a2"

    # Metadata preserved
    assert loaded.metadata["goal"] == "roundtrip test"
    assert loaded.metadata["extra"] == 42

    # Node attributes preserved
    a1_loaded = next(n for n in loaded.nodes if n.node_id == "a1")
    assert a1_loaded.status == NodeStatus.ATOMIC
    assert a1_loaded.type_signature == "nat -> nat"
    assert len(a1_loaded.inputs) == 1
    assert a1_loaded.inputs[0].name == "x"


# ---------------------------------------------------------------------------
# Test 2: model_dump / model_validate roundtrip
# ---------------------------------------------------------------------------


def test_cdg_model_dump_roundtrip() -> None:
    """model_dump() -> model_validate() preserves structure."""
    root = _make_root_node(children=["a1"])
    a1 = _make_atomic_node("a1", "Atom A")
    edge = _make_edge("a1", "a1")  # self-edge is fine for roundtrip test

    cdg = CDGExport(
        nodes=[root, a1],
        edges=[edge],
        metadata={"key": "value"},
    )

    dumped = cdg.model_dump()
    restored = CDGExport.model_validate(dumped)

    assert len(restored.nodes) == len(cdg.nodes)
    assert len(restored.edges) == len(cdg.edges)
    assert restored.metadata == cdg.metadata

    for orig, rest in zip(cdg.nodes, restored.nodes):
        assert orig.node_id == rest.node_id
        assert orig.name == rest.name
        assert orig.status == rest.status

    assert restored.edges[0].source_id == cdg.edges[0].source_id


# ---------------------------------------------------------------------------
# Test 3: result_to_cdg preserves structure (3 leaves, 2 matched)
# ---------------------------------------------------------------------------


def test_result_to_cdg_preserves_structure() -> None:
    """3 atomic leaves, 2 matched. Assert correct node count and coverage == 2/3."""
    root = _make_root_node(children=["a1", "a2", "a3"])
    a1 = _make_atomic_node("a1", "Atom A")
    a2 = _make_atomic_node("a2", "Atom B")
    a3 = _make_atomic_node("a3", "Atom C")

    cdg = CDGExport(nodes=[root, a1, a2, a3], edges=[])
    match_results = [
        _make_match_result("a1", success=True),
        _make_match_result("a2", success=True),
        _make_match_result("a3", success=False),
    ]
    result = OrchestratorResult(cdg=cdg, match_results=match_results, rounds_used=1)
    metadata = _make_metadata()

    cdg_dict = orchestrator_result_to_cdg(result, metadata)

    # 4 nodes total (root + 3 leaves)
    assert len(cdg_dict["nodes"]) == 4
    node_ids = {n["node_id"] for n in cdg_dict["nodes"]}
    assert node_ids == {"root", "a1", "a2", "a3"}

    # Coverage: 2 of 3 atomic nodes matched
    assert abs(metadata.verified_leaf_coverage - 2 / 3) < 1e-9


# ---------------------------------------------------------------------------
# Test 4: sanitize_cdg is idempotent
# ---------------------------------------------------------------------------


def test_result_to_cdg_sanitize_idempotent() -> None:
    """Running sanitize_cdg() twice produces the same result."""
    root = _make_root_node(children=["a1", "a2"])
    a1 = _make_atomic_node("a1", "Atom A")
    a2 = _make_atomic_node("a2", "Atom B")

    cdg = CDGExport(nodes=[root, a1, a2], edges=[_make_edge("a1", "a2")])
    match_results = [
        _make_match_result("a1", success=True),
        _make_match_result("a2", success=False),
    ]
    result = OrchestratorResult(cdg=cdg, match_results=match_results, rounds_used=1)
    metadata = _make_metadata()

    cdg_dict = orchestrator_result_to_cdg(result, metadata)

    # Apply sanitize_cdg a second time
    sanitized_again = sanitize_cdg(cdg_dict)

    assert sanitized_again["nodes"] == cdg_dict["nodes"]
    assert sanitized_again["edges"] == cdg_dict["edges"]


# ---------------------------------------------------------------------------
# Test 5: All exemplar JSON files are valid CDG structures (parametrized)
# ---------------------------------------------------------------------------

_exemplar_files = sorted(EXEMPLAR_DIR.glob("*.json"))


@pytest.mark.parametrize(
    "exemplar_path",
    _exemplar_files,
    ids=[p.stem for p in _exemplar_files],
)
def test_exemplar_json_is_valid_cdg(exemplar_path: Path) -> None:
    """Parse each exemplar, assert ATOMIC leaves exist, edges exist,
    and if a root node is present it has status 'decomposed'."""
    with open(exemplar_path) as f:
        data = json.load(f)

    assert "nodes" in data, f"{exemplar_path.name} missing 'nodes' key"
    assert "edges" in data, f"{exemplar_path.name} missing 'edges' key"

    nodes = data["nodes"]
    edges = data["edges"]

    assert len(nodes) > 0, f"{exemplar_path.name} has no nodes"
    assert len(edges) > 0, f"{exemplar_path.name} has no edges"

    # At least one ATOMIC leaf must exist
    atomic_nodes = [n for n in nodes if n.get("status") == "atomic"]
    assert len(atomic_nodes) > 0, f"{exemplar_path.name} has no atomic leaves"

    # If a root node (parent_id is None) exists, it must be decomposed
    root_nodes = [n for n in nodes if n.get("parent_id") is None]
    for root in root_nodes:
        assert root.get("status") == "decomposed", (
            f"{exemplar_path.name}: root node '{root.get('node_id')}' "
            f"has status '{root.get('status')}', expected 'decomposed'"
        )


# ---------------------------------------------------------------------------
# Test 6: beta_bernoulli.json has 3 atomic nodes
# ---------------------------------------------------------------------------


def test_conjugate_exemplar_has_3_atomic_nodes() -> None:
    """Load beta_bernoulli.json, assert 3 atomic nodes."""
    path = EXEMPLAR_DIR / "beta_bernoulli.json"
    with open(path) as f:
        data = json.load(f)

    atomic = [n for n in data["nodes"] if n.get("status") == "atomic"]
    assert len(atomic) == 3


# ---------------------------------------------------------------------------
# Test 7: signal_event_rate.json has 3-stage linear pipeline
# ---------------------------------------------------------------------------


def test_signal_event_rate_exemplar_has_3_stage_pipeline() -> None:
    """Load signal_event_rate.json, assert 3 atomic nodes, 2 edges,
    linear pipeline structure, and metadata.verified_leaf_coverage == 1.0."""
    path = EXEMPLAR_DIR / "signal_event_rate.json"
    with open(path) as f:
        data = json.load(f)

    nodes = data["nodes"]
    edges = data["edges"]

    # 3 atomic nodes
    atomic = [n for n in nodes if n.get("status") == "atomic"]
    assert len(atomic) == 3

    # 2 edges
    assert len(edges) == 2

    # Linear pipeline: edges form a chain A -> B -> C
    source_ids = [e["source_id"] for e in edges]
    target_ids = [e["target_id"] for e in edges]

    # Each node appears at most once as source and once as target
    assert len(set(source_ids)) == 2
    assert len(set(target_ids)) == 2

    # The head of the pipeline is in sources but not in targets
    heads = set(source_ids) - set(target_ids)
    assert len(heads) == 1, "Pipeline should have exactly one head node"

    # The tail is in targets but not in sources
    tails = set(target_ids) - set(source_ids)
    assert len(tails) == 1, "Pipeline should have exactly one tail node"

    # Verified leaf coverage
    assert data["metadata"]["verified_leaf_coverage"] == 1.0

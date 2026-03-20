"""E2E tests: synthesizer produces valid Python for non-signal domains.

Verifies that the Assembler can build parseable, sorry-free Python
skeletons for graph-algorithm, linear-algebra, and sorting CDGs --
domains that have no relationship to the signal-processing aliases
hard-coded in the pipeline template.
"""

from __future__ import annotations

import ast

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.synthesizer.assembler import Assembler, sanitize_name
from sciona.synthesizer.python_template import generate_pipeline_py
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


def _make_match_result(node: AlgorithmicNode, decl_name: str) -> MatchResult:
    """Build a successful MatchResult for an ATOMIC node."""
    type_sig = node.type_signature or "Any -> Any"
    decl = Declaration(
        name=decl_name,
        type_signature=type_sig,
        prover=Prover.PYTHON,
    )
    candidate = CandidateMatch(
        declaration=decl, score=1.0, retrieval_method="test_mock"
    )
    vr = VerificationResult(
        candidate=candidate, verified=True, proof_term=decl_name
    )
    return MatchResult(
        pdg_node=PDGNode(
            predicate_id=node.node_id,
            statement=type_sig,
            prover=Prover.PYTHON,
        ),
        verified_match=vr,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


# ---------------------------------------------------------------------------
# Graph-algorithm CDG (3 nodes: Initialize Distances, Relax Edges, Extract Path)
# ---------------------------------------------------------------------------


def _graph_algo_cdg() -> tuple[CDGExport, list[MatchResult]]:
    root = AlgorithmicNode(
        node_id="graph_root",
        name="Shortest Path Solver",
        description="Find shortest path in a weighted graph",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.DECOMPOSED,
        children=["graph_n1", "graph_n2", "graph_n3"],
        depth=0,
        type_signature="(graph: dict, source: int) -> list[int]",
        inputs=[
            IOSpec(name="graph", type_desc="dict"),
            IOSpec(name="source", type_desc="int"),
        ],
        outputs=[IOSpec(name="path", type_desc="list[int]")],
    )

    n1 = AlgorithmicNode(
        node_id="graph_n1",
        parent_id="graph_root",
        name="Initialize Distances",
        description="Set all distances to infinity except source",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(graph: dict, source: int) -> dict[int, float]",
        matched_primitive="graph_utils.init_distances",
        inputs=[
            IOSpec(name="graph", type_desc="dict"),
            IOSpec(name="source", type_desc="int"),
        ],
        outputs=[IOSpec(name="distances", type_desc="dict[int, float]")],
    )

    n2 = AlgorithmicNode(
        node_id="graph_n2",
        parent_id="graph_root",
        name="Relax Edges",
        description="Relax all edges repeatedly to find shortest distances",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(graph: dict, distances: dict[int, float]) -> dict[int, float]",
        matched_primitive="graph_utils.relax_edges",
        inputs=[
            IOSpec(name="graph", type_desc="dict"),
            IOSpec(name="distances", type_desc="dict[int, float]"),
        ],
        outputs=[IOSpec(name="relaxed", type_desc="dict[int, float]")],
    )

    n3 = AlgorithmicNode(
        node_id="graph_n3",
        parent_id="graph_root",
        name="Extract Path",
        description="Reconstruct shortest path from distance map",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(distances: dict[int, float], source: int) -> list[int]",
        matched_primitive="graph_utils.extract_path",
        inputs=[
            IOSpec(name="distances", type_desc="dict[int, float]"),
            IOSpec(name="source", type_desc="int"),
        ],
        outputs=[IOSpec(name="path", type_desc="list[int]")],
    )

    edges = [
        DependencyEdge(
            source_id="graph_n1",
            target_id="graph_n2",
            output_name="distances",
            input_name="distances",
            source_type="dict[int, float]",
            target_type="dict[int, float]",
        ),
        DependencyEdge(
            source_id="graph_n2",
            target_id="graph_n3",
            output_name="relaxed",
            input_name="distances",
            source_type="dict[int, float]",
            target_type="dict[int, float]",
        ),
    ]

    cdg = CDGExport(
        nodes=[root, n1, n2, n3],
        edges=edges,
        metadata={"goal": "Shortest path via edge relaxation"},
    )

    match_results = [
        _make_match_result(n1, "graph_utils.init_distances"),
        _make_match_result(n2, "graph_utils.relax_edges"),
        _make_match_result(n3, "graph_utils.extract_path"),
    ]

    return cdg, match_results


# ---------------------------------------------------------------------------
# Linear-algebra CDG (2 nodes: Factor Matrix, Back Solve)
# ---------------------------------------------------------------------------


def _linalg_cdg() -> tuple[CDGExport, list[MatchResult]]:
    root = AlgorithmicNode(
        node_id="linalg_root",
        name="Linear System Solver",
        description="Solve Ax = b via factorization",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.DECOMPOSED,
        children=["linalg_n1", "linalg_n2"],
        depth=0,
        type_signature="(A: ndarray, b: ndarray) -> ndarray",
        inputs=[
            IOSpec(name="A", type_desc="ndarray"),
            IOSpec(name="b", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="x", type_desc="ndarray")],
    )

    n1 = AlgorithmicNode(
        node_id="linalg_n1",
        parent_id="linalg_root",
        name="Factor Matrix",
        description="Compute LU factorization of A",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(A: ndarray) -> tuple[ndarray, ndarray]",
        matched_primitive="scipy.linalg.lu_factor",
        inputs=[IOSpec(name="A", type_desc="ndarray")],
        outputs=[IOSpec(name="lu_piv", type_desc="tuple[ndarray, ndarray]")],
    )

    n2 = AlgorithmicNode(
        node_id="linalg_n2",
        parent_id="linalg_root",
        name="Back Solve",
        description="Solve triangular system given LU factors",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(lu_piv: tuple[ndarray, ndarray], b: ndarray) -> ndarray",
        matched_primitive="scipy.linalg.lu_solve",
        inputs=[
            IOSpec(name="lu_piv", type_desc="tuple[ndarray, ndarray]"),
            IOSpec(name="b", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="x", type_desc="ndarray")],
    )

    edges = [
        DependencyEdge(
            source_id="linalg_n1",
            target_id="linalg_n2",
            output_name="lu_piv",
            input_name="lu_piv",
            source_type="tuple[ndarray, ndarray]",
            target_type="tuple[ndarray, ndarray]",
        ),
    ]

    cdg = CDGExport(
        nodes=[root, n1, n2],
        edges=edges,
        metadata={"goal": "Solve linear system via LU factorization"},
    )

    match_results = [
        _make_match_result(n1, "scipy.linalg.lu_factor"),
        _make_match_result(n2, "scipy.linalg.lu_solve"),
    ]

    return cdg, match_results


# ---------------------------------------------------------------------------
# Sorting CDG (2 nodes: Partition, Merge)
# ---------------------------------------------------------------------------


def _sorting_cdg() -> tuple[CDGExport, list[MatchResult]]:
    root = AlgorithmicNode(
        node_id="sort_root",
        name="Sort Array",
        description="Sort an array using divide and conquer",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.DECOMPOSED,
        children=["sort_n1", "sort_n2"],
        depth=0,
        type_signature="(arr: list[int]) -> list[int]",
        inputs=[IOSpec(name="arr", type_desc="list[int]")],
        outputs=[IOSpec(name="sorted_arr", type_desc="list[int]")],
    )

    n1 = AlgorithmicNode(
        node_id="sort_n1",
        parent_id="sort_root",
        name="Partition",
        description="Partition the array around a pivot",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(arr: list[int]) -> tuple[list[int], list[int]]",
        matched_primitive="sort_utils.partition",
        inputs=[IOSpec(name="arr", type_desc="list[int]")],
        outputs=[
            IOSpec(name="parts", type_desc="tuple[list[int], list[int]]"),
        ],
    )

    n2 = AlgorithmicNode(
        node_id="sort_n2",
        parent_id="sort_root",
        name="Merge",
        description="Merge two sorted halves",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        depth=1,
        type_signature="(parts: tuple[list[int], list[int]]) -> list[int]",
        matched_primitive="sort_utils.merge",
        inputs=[
            IOSpec(name="parts", type_desc="tuple[list[int], list[int]]"),
        ],
        outputs=[IOSpec(name="sorted_arr", type_desc="list[int]")],
    )

    edges = [
        DependencyEdge(
            source_id="sort_n1",
            target_id="sort_n2",
            output_name="parts",
            input_name="parts",
            source_type="tuple[list[int], list[int]]",
            target_type="tuple[list[int], list[int]]",
        ),
    ]

    cdg = CDGExport(
        nodes=[root, n1, n2],
        edges=edges,
        metadata={"goal": "Sort array via partition and merge"},
    )

    match_results = [
        _make_match_result(n1, "sort_utils.partition"),
        _make_match_result(n2, "sort_utils.merge"),
    ]

    return cdg, match_results


# =========================================================================
# Tests
# =========================================================================


class TestCrossDomainSynthesis:
    """Verify the synthesizer produces valid Python for non-signal domains."""

    # 1. Graph-algorithm skeleton is valid Python
    def test_graph_algo_skeleton_is_valid_python(self):
        cdg, match_results = _graph_algo_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)

        try:
            ast.parse(skeleton.source_code)
        except SyntaxError as exc:
            pytest.fail(
                f"Graph-algo skeleton has syntax error: {exc}\n\n"
                f"--- source ---\n{skeleton.source_code}"
            )

        assert skeleton.sorry_count == 0, (
            f"Expected 0 sorry stubs, got {skeleton.sorry_count}"
        )

    # 2. Linalg skeleton is valid Python
    def test_linalg_skeleton_is_valid_python(self):
        cdg, match_results = _linalg_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)

        try:
            ast.parse(skeleton.source_code)
        except SyntaxError as exc:
            pytest.fail(
                f"Linalg skeleton has syntax error: {exc}\n\n"
                f"--- source ---\n{skeleton.source_code}"
            )

        assert skeleton.sorry_count == 0, (
            f"Expected 0 sorry stubs, got {skeleton.sorry_count}"
        )

    # 3. Sorting skeleton is valid Python
    def test_sorting_skeleton_is_valid_python(self):
        cdg, match_results = _sorting_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)

        try:
            ast.parse(skeleton.source_code)
        except SyntaxError as exc:
            pytest.fail(
                f"Sorting skeleton has syntax error: {exc}\n\n"
                f"--- source ---\n{skeleton.source_code}"
            )

        assert skeleton.sorry_count == 0, (
            f"Expected 0 sorry stubs, got {skeleton.sorry_count}"
        )

    # 4. No signal alias in graph-algo skeleton
    def test_no_signal_alias_in_graph_algo_skeleton(self):
        cdg, match_results = _graph_algo_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)
        source_lower = skeleton.source_code.lower()

        assert "ecg" not in source_lower, (
            "'ecg' found in graph-algo skeleton source"
        )
        assert "ppg" not in source_lower, (
            "'ppg' found in graph-algo skeleton source"
        )

    # 5. No signal alias in linalg skeleton
    def test_no_signal_alias_in_linalg_skeleton(self):
        cdg, match_results = _linalg_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)
        source_lower = skeleton.source_code.lower()

        assert "ecg" not in source_lower, (
            "'ecg' found in linalg skeleton source"
        )
        assert "ppg" not in source_lower, (
            "'ppg' found in linalg skeleton source"
        )

    # 6. Composition function named correctly
    def test_composition_function_named_correctly(self):
        cdg, match_results = _graph_algo_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)

        # Root is "Shortest Path Solver" -> sanitize_name -> "shortest_path_solver"
        expected_fn = sanitize_name("Shortest Path Solver") + "_composition"
        assert expected_fn in skeleton.source_code, (
            f"Expected composition function '{expected_fn}' not found in source. "
            f"Source excerpt:\n{skeleton.source_code[:500]}"
        )

    # 7. generate_pipeline_py output is valid Python
    def test_pipeline_py_no_signal_aliases(self):
        pipeline_source = generate_pipeline_py(
            pipeline_steps=[],
            entrypoint_names=["run"],
            default_entrypoint="run",
        )

        try:
            ast.parse(pipeline_source)
        except SyntaxError as exc:
            pytest.fail(
                f"pipeline.py has syntax error: {exc}\n\n"
                f"--- source ---\n{pipeline_source}"
            )

    # 8. Skeleton units match CDG leaves
    def test_skeleton_units_match_cdg_leaves(self):
        cdg, match_results = _graph_algo_cdg()
        skeleton = Assembler(Prover.PYTHON).assemble(cdg, match_results)

        leaf_ids = {n.node_id for n in cdg.leaf_nodes()}
        unit_ids = {u.node_id for u in skeleton.units}

        assert unit_ids == leaf_ids, (
            f"Skeleton unit node_ids {unit_ids} do not match "
            f"CDG leaf node_ids {leaf_ids}"
        )

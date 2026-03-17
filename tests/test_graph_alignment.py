"""Tests for GraphAlignmentScorer."""

from __future__ import annotations

import pytest

from ageom.architect.graph_alignment import AlignmentScore, GraphAlignmentScorer
from ageom.architect.graph_retrieval import (
    ExampleChild,
    ExampleDecomposition,
    ExampleEdge,
)
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_node(
    concept_type: ConceptType = ConceptType.SORTING,
    n_inputs: int = 2,
    n_outputs: int = 1,
    type_signature: str = "",
    **kwargs,
) -> AlgorithmicNode:
    inputs = [IOSpec(name=f"in{i}", type_desc="int") for i in range(n_inputs)]
    outputs = [IOSpec(name=f"out{i}", type_desc="int") for i in range(n_outputs)]
    return AlgorithmicNode(
        node_id="q-root",
        name="query",
        description="query node",
        concept_type=concept_type,
        inputs=inputs,
        outputs=outputs,
        type_signature=type_signature,
        **kwargs,
    )


def _make_child(
    concept_type: ConceptType = ConceptType.SORTING,
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=f"child-{concept_type.value}",
        parent_id="q-root",
        name=f"child-{concept_type.value}",
        description="child",
        concept_type=concept_type,
        type_signature=type_signature,
    )


def _make_example_child(
    concept_type: str = "sorting",
    witness_input_types: list[str] | None = None,
    witness_output_types: list[str] | None = None,
) -> ExampleChild:
    return ExampleChild(
        node_id=f"ec-{concept_type}",
        name=f"ex-child-{concept_type}",
        description="example child",
        concept_type=concept_type,
        status="atomic",
        n_inputs=1,
        n_outputs=1,
        type_signature="",
        witness_input_types=witness_input_types or [],
        witness_output_types=witness_output_types or [],
    )


def _make_candidate(
    concept_type: str = "sorting",
    n_inputs: int = 2,
    n_outputs: int = 1,
    children: list[ExampleChild] | None = None,
    edges: list[ExampleEdge] | None = None,
    abstract_type_class: str = "",
) -> ExampleDecomposition:
    return ExampleDecomposition(
        fqn="repo.module.func",
        name="candidate",
        description="candidate decomposition",
        concept_type=concept_type,
        repo="test-repo",
        topo_hash="abc123",
        children=children or [],
        edges=edges or [],
        retrieval_layer=1,
        score=0.9,
        abstract_type_class=abstract_type_class,
        n_inputs=n_inputs,
        n_outputs=n_outputs,
    )


def _make_query_edge(src: str, tgt: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out0",
        input_name="in0",
        source_type="int",
        target_type="int",
    )


def _make_candidate_edge(src: str, tgt: str) -> ExampleEdge:
    return ExampleEdge(
        source_id=src,
        target_id=tgt,
        output_name="out0",
        input_name="in0",
    )


# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------


_scorer = GraphAlignmentScorer()


@pytest.mark.parametrize(
    "test_id, query_node, query_children, query_edges, candidate, expected_checks",
    [
        pytest.param(
            "exact_match",
            _make_node(ConceptType.SORTING, n_inputs=2, n_outputs=1),
            [_make_child(ConceptType.SEARCHING), _make_child(ConceptType.ARITHMETIC)],
            [_make_query_edge("a", "b"), _make_query_edge("b", "c")],
            _make_candidate(
                concept_type="sorting",
                n_inputs=2,
                n_outputs=1,
                children=[
                    _make_example_child("searching"),
                    _make_example_child("arithmetic"),
                ],
                edges=[
                    _make_candidate_edge("x", "y"),
                    _make_candidate_edge("y", "z"),
                ],
            ),
            {
                "concept_type_match": 1.0,
                "io_arity_match": 1.0,
                "child_concept_overlap": 1.0,
                "topo_match": 1.0,
                "type_class_match": 1.0,
                "witness_type_match": 1.0,
                "total_ge": 0.99,
            },
            id="exact_match",
        ),
        pytest.param(
            "partial_match",
            _make_node(ConceptType.SORTING, n_inputs=3, n_outputs=1),
            [_make_child(ConceptType.SEARCHING), _make_child(ConceptType.GREEDY)],
            [_make_query_edge("a", "b")],
            _make_candidate(
                concept_type="sorting",
                n_inputs=2,
                n_outputs=1,
                children=[
                    _make_example_child("searching"),
                    _make_example_child("arithmetic"),
                ],
                edges=[
                    _make_candidate_edge("x", "y"),
                    _make_candidate_edge("y", "z"),
                ],
            ),
            {
                "concept_type_match": 1.0,
                "io_arity_match": 0.85,
                "child_concept_overlap_lt": 1.0,
                "child_concept_overlap_gt": 0.0,
                "topo_match_lt": 1.0,
                "total_gt": 0.3,
                "total_lt": 1.0,
            },
            id="partial_match",
        ),
        pytest.param(
            "no_match",
            _make_node(ConceptType.GRAPH_TRAVERSAL, n_inputs=1, n_outputs=3),
            [_make_child(ConceptType.GRAPH_OPTIMIZATION)],
            [],
            _make_candidate(
                concept_type="sorting",
                n_inputs=2,
                n_outputs=1,
                children=[_make_example_child("arithmetic")],
                edges=[_make_candidate_edge("x", "y")],
            ),
            {
                "concept_type_match": 0.0,
                "child_concept_overlap": 0.0,
                "topo_match": 0.0,
                "total_lt": 0.5,
            },
            id="no_match",
        ),
        pytest.param(
            "missing_metadata",
            _make_node(ConceptType.SORTING, n_inputs=2, n_outputs=1),
            [_make_child(ConceptType.SEARCHING)],
            [],
            _make_candidate(
                concept_type="sorting",
                n_inputs=0,
                n_outputs=0,
                children=[_make_example_child("searching")],
                edges=[],
            ),
            {
                "concept_type_match": 1.0,
                "io_arity_match": 1.0,  # fallback for missing metadata
                "child_concept_overlap": 1.0,
                "topo_match": 1.0,  # both empty
                "type_class_match": 1.0,
                "witness_type_match": 1.0,
                "total_ge": 0.99,
            },
            id="missing_metadata",
        ),
    ],
)
def test_graph_alignment_scorer(
    test_id, query_node, query_children, query_edges, candidate, expected_checks
):
    result = _scorer.score(query_node, query_children, query_edges, candidate)

    for key, value in expected_checks.items():
        if key == "total_ge":
            assert result.total >= value, f"{test_id}: total {result.total} < {value}"
        elif key == "total_gt":
            assert result.total > value, f"{test_id}: total {result.total} <= {value}"
        elif key == "total_lt":
            assert result.total < value, f"{test_id}: total {result.total} >= {value}"
        elif key.endswith("_lt"):
            field_name = key[: -len("_lt")]
            actual = getattr(result, field_name)
            assert actual < value, f"{test_id}: {field_name} {actual} >= {value}"
        elif key.endswith("_gt"):
            field_name = key[: -len("_gt")]
            actual = getattr(result, field_name)
            assert actual > value, f"{test_id}: {field_name} {actual} <= {value}"
        elif key.endswith("_ge"):
            field_name = key[: -len("_ge")]
            actual = getattr(result, field_name)
            assert actual >= value, f"{test_id}: {field_name} {actual} < {value}"
        else:
            actual = getattr(result, key)
            assert actual == pytest.approx(
                value, abs=1e-6
            ), f"{test_id}: {key} expected {value}, got {actual}"


def test_witness_type_matching():
    """Verify witness type overlap is computed correctly."""
    query_node = _make_node(ConceptType.SORTING)
    query_children = [
        _make_child(ConceptType.SEARCHING, type_signature="list[int] -> int"),
        _make_child(ConceptType.ARITHMETIC, type_signature="int -> int"),
    ]
    candidate = _make_candidate(
        concept_type="sorting",
        children=[
            _make_example_child(
                "searching", witness_input_types=["list[int] -> int"]
            ),
            _make_example_child("arithmetic", witness_output_types=["float -> float"]),
        ],
    )

    result = _scorer.score(query_node, query_children, [], candidate)
    # One of two children has a matching type_signature
    assert result.witness_type_match == pytest.approx(0.5)


def test_type_class_mismatch():
    """Verify type_class_match is 0 when query has a class that doesn't match."""
    query_node = _make_node(ConceptType.SORTING)
    # Attach abstract_type_class via object.__setattr__ since it's not a model field
    object.__setattr__(query_node, "abstract_type_class", "Monad")
    candidate = _make_candidate(
        concept_type="sorting",
        abstract_type_class="Functor",
    )

    result = _scorer.score(query_node, [], [], candidate)
    assert result.type_class_match == 0.0


def test_total_is_weighted_sum():
    """Verify total equals the documented weighted sum."""
    query_node = _make_node(ConceptType.SORTING, n_inputs=2, n_outputs=1)
    candidate = _make_candidate(concept_type="sorting", n_inputs=2, n_outputs=1)

    result = _scorer.score(query_node, [], [], candidate)

    expected_total = (
        0.25 * result.concept_type_match
        + 0.15 * result.io_arity_match
        + 0.25 * result.child_concept_overlap
        + 0.15 * result.topo_match
        + 0.10 * result.type_class_match
        + 0.10 * result.witness_type_match
    )
    assert result.total == pytest.approx(expected_total, abs=1e-9)

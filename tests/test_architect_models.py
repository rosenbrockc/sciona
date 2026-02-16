"""Tests for ageom.architect.models — CDG Pydantic models."""

import pytest

from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
    SkeletonGraph,
)


class TestConceptType:
    def test_all_values_are_strings(self):
        for ct in ConceptType:
            assert isinstance(ct.value, str)

    def test_expected_members(self):
        expected = {
            "sorting", "searching", "divide_and_conquer", "greedy",
            "dynamic_programming", "graph_traversal", "graph_optimization",
            "string_matching", "geometry", "arithmetic", "number_theory",
            "combinatorics", "algebra", "analysis", "set_theory", "custom",
        }
        assert {ct.value for ct in ConceptType} == expected


class TestIOSpec:
    def test_basic_creation(self):
        io = IOSpec(name="arr", type_desc="list[int]")
        assert io.name == "arr"
        assert io.type_desc == "list[int]"
        assert io.constraints == ""

    def test_with_constraints(self):
        io = IOSpec(name="arr", type_desc="list[int]", constraints="sorted, non-empty")
        assert io.constraints == "sorted, non-empty"

    def test_missing_required_field(self):
        with pytest.raises(Exception):
            IOSpec(name="x")  # type: ignore[call-arg]


class TestNodeStatus:
    def test_values(self):
        assert NodeStatus.PENDING == "pending"
        assert NodeStatus.DECOMPOSED == "decomposed"
        assert NodeStatus.ATOMIC == "atomic"
        assert NodeStatus.REJECTED == "rejected"
        assert NodeStatus.HIGH_RISK == "high_risk"


class TestAlgorithmicNode:
    def test_minimal_creation(self):
        node = AlgorithmicNode(
            node_id="n1",
            name="Sort Array",
            description="Sort an array of integers",
            concept_type=ConceptType.SORTING,
        )
        assert node.node_id == "n1"
        assert node.parent_id is None
        assert node.status == NodeStatus.PENDING
        assert node.children == []
        assert node.depth == 0
        assert node.matched_primitive is None

    def test_full_creation(self):
        node = AlgorithmicNode(
            node_id="n2",
            parent_id="n1",
            name="Merge Step",
            description="Merge two sorted halves",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            inputs=[IOSpec(name="left", type_desc="list[int]")],
            outputs=[IOSpec(name="merged", type_desc="list[int]")],
            status=NodeStatus.ATOMIC,
            children=["n3", "n4"],
            depth=2,
            type_signature="list[int] -> list[int] -> list[int]",
            matched_primitive="merge",
            critic_notes="Well-defined",
            decomposition_rationale="Standard merge operation",
        )
        assert node.parent_id == "n1"
        assert node.status == NodeStatus.ATOMIC
        assert len(node.inputs) == 1
        assert len(node.outputs) == 1
        assert node.children == ["n3", "n4"]

    def test_serialization_roundtrip(self):
        node = AlgorithmicNode(
            node_id="n1",
            name="Test",
            description="desc",
            concept_type=ConceptType.GREEDY,
            inputs=[IOSpec(name="x", type_desc="int")],
        )
        data = node.model_dump()
        restored = AlgorithmicNode.model_validate(data)
        assert restored == node


class TestDependencyEdge:
    def test_creation(self):
        edge = DependencyEdge(
            source_id="n1",
            target_id="n2",
            output_name="sorted",
            input_name="data",
            source_type="list[int]",
            target_type="list[int]",
        )
        assert edge.source_id == "n1"
        assert edge.target_id == "n2"
        assert edge.requires_glue is False

    def test_type_mismatch_with_glue(self):
        edge = DependencyEdge(
            source_id="n1",
            target_id="n2",
            output_name="result",
            input_name="data",
            source_type="list[float]",
            target_type="list[int]",
            requires_glue=True,
        )
        assert edge.requires_glue is True

    def test_serialization_roundtrip(self):
        edge = DependencyEdge(
            source_id="a",
            target_id="b",
            output_name="out",
            input_name="in",
            source_type="str",
            target_type="str",
        )
        data = edge.model_dump()
        restored = DependencyEdge.model_validate(data)
        assert restored == edge


class TestSkeletonGraph:
    def test_creation(self):
        sg = SkeletonGraph(
            paradigm=ConceptType.SORTING,
            name="Sorting",
            description="Compare-swap paradigm",
        )
        assert sg.template_nodes == []
        assert sg.template_edges == []
        assert sg.variants == []

    def test_with_nodes_and_edges(self):
        n1 = AlgorithmicNode(
            node_id="compare",
            name="Compare",
            description="Compare two elements",
            concept_type=ConceptType.SORTING,
        )
        n2 = AlgorithmicNode(
            node_id="swap",
            name="Swap",
            description="Swap elements",
            concept_type=ConceptType.SORTING,
        )
        edge = DependencyEdge(
            source_id="compare",
            target_id="swap",
            output_name="order",
            input_name="i",
            source_type="bool",
            target_type="int",
        )
        sg = SkeletonGraph(
            paradigm=ConceptType.SORTING,
            name="Sorting",
            description="desc",
            template_nodes=[n1, n2],
            template_edges=[edge],
            variants=["insertion_sort", "heapsort"],
        )
        assert len(sg.template_nodes) == 2
        assert len(sg.template_edges) == 1
        assert "insertion_sort" in sg.variants


class TestAlgorithmicPrimitive:
    def test_creation(self):
        prim = AlgorithmicPrimitive(
            name="heapsort",
            source="clrs-30",
            category=ConceptType.SORTING,
            description="Heapsort algorithm",
        )
        assert prim.name == "heapsort"
        assert prim.source == "clrs-30"
        assert prim.inputs == []
        assert prim.clrs_spec == {}

    def test_with_io(self):
        prim = AlgorithmicPrimitive(
            name="dijkstra",
            source="clrs-30",
            category=ConceptType.GRAPH_OPTIMIZATION,
            description="Single-source shortest paths",
            inputs=[
                IOSpec(name="graph", type_desc="weighted Graph"),
                IOSpec(name="source", type_desc="node"),
            ],
            outputs=[
                IOSpec(name="distances", type_desc="dict[node, float]"),
            ],
            type_signature="Graph -> Node -> Dict[Node, Float]",
            clrs_spec={"adj": "(input, node, pointer)"},
        )
        assert len(prim.inputs) == 2
        assert len(prim.outputs) == 1
        assert prim.type_signature != ""

    def test_serialization_roundtrip(self):
        prim = AlgorithmicPrimitive(
            name="test",
            source="test",
            category=ConceptType.CUSTOM,
            description="test prim",
        )
        data = prim.model_dump()
        restored = AlgorithmicPrimitive.model_validate(data)
        assert restored == prim

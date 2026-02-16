"""Tests for ageom.architect.handoff — CDG serialization and Round 2 conversion."""

import json
import pytest
from pathlib import Path

from ageom.architect.handoff import (
    CDGExport,
    HandoffValidationError,
    export_cdg,
    load_json,
    save_json,
    to_pdg_nodes,
)
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.types import Prover


@pytest.fixture
def atomic_nodes() -> list[AlgorithmicNode]:
    """A simple two-node CDG where both leaves are atomic."""
    return [
        AlgorithmicNode(
            node_id="root",
            name="Sort and Search",
            description="Sort an array then binary search",
            concept_type=ConceptType.DIVIDE_AND_CONQUER,
            status=NodeStatus.DECOMPOSED,
            children=["sort_step", "search_step"],
            depth=0,
        ),
        AlgorithmicNode(
            node_id="sort_step",
            parent_id="root",
            name="Heapsort",
            description="Sort array using heapsort",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.ATOMIC,
            matched_primitive="heapsort",
            type_signature="list[int] -> list[int]",
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
            depth=1,
        ),
        AlgorithmicNode(
            node_id="search_step",
            parent_id="root",
            name="Binary Search",
            description="Search sorted array for target",
            concept_type=ConceptType.SEARCHING,
            status=NodeStatus.ATOMIC,
            matched_primitive="binary_search",
            type_signature="list[int] -> int -> int",
            inputs=[IOSpec(name="arr", type_desc="sorted list[int]"),
                    IOSpec(name="target", type_desc="int")],
            outputs=[IOSpec(name="index", type_desc="int")],
            depth=1,
        ),
    ]


@pytest.fixture
def edges() -> list[DependencyEdge]:
    return [
        DependencyEdge(
            source_id="sort_step",
            target_id="search_step",
            output_name="sorted",
            input_name="arr",
            source_type="list[int]",
            target_type="sorted list[int]",
        ),
    ]


@pytest.fixture
def non_atomic_nodes() -> list[AlgorithmicNode]:
    """CDG with a non-atomic leaf (should fail validation)."""
    return [
        AlgorithmicNode(
            node_id="root",
            name="Complex Task",
            description="A complex task",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.DECOMPOSED,
            children=["step1"],
            depth=0,
        ),
        AlgorithmicNode(
            node_id="step1",
            parent_id="root",
            name="Unresolved Step",
            description="This step is not yet decomposed",
            concept_type=ConceptType.CUSTOM,
            status=NodeStatus.PENDING,  # NOT atomic!
            depth=1,
        ),
    ]


class TestCDGExport:
    def test_leaf_nodes(self, atomic_nodes, edges):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        leaves = cdg.leaf_nodes()
        assert len(leaves) == 2
        leaf_names = {n.name for n in leaves}
        assert leaf_names == {"Heapsort", "Binary Search"}

    def test_is_complete_true(self, atomic_nodes, edges):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        assert cdg.is_complete() is True

    def test_is_complete_false(self, non_atomic_nodes):
        cdg = CDGExport(nodes=non_atomic_nodes, edges=[])
        assert cdg.is_complete() is False

    def test_non_atomic_leaves(self, non_atomic_nodes):
        cdg = CDGExport(nodes=non_atomic_nodes, edges=[])
        non_atomic = cdg.non_atomic_leaves()
        assert len(non_atomic) == 1
        assert non_atomic[0].name == "Unresolved Step"


class TestExportCDG:
    def test_valid_export(self, atomic_nodes, edges):
        cdg = export_cdg(
            atomic_nodes, edges,
            goal="Sort and search",
            paradigm="divide_and_conquer",
        )
        assert cdg.metadata["goal"] == "Sort and search"
        assert cdg.metadata["num_nodes"] == 3
        assert cdg.metadata["num_edges"] == 1
        assert "timestamp" in cdg.metadata

    def test_rejects_non_atomic_leaves(self, non_atomic_nodes):
        with pytest.raises(ValueError, match="non-atomic leaf"):
            export_cdg(non_atomic_nodes, [])


class TestToPDGNodes:
    def test_converts_atomic_leaves(self, atomic_nodes, edges):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        pdg_nodes = to_pdg_nodes(cdg, prover=Prover.LEAN4)

        assert len(pdg_nodes) == 2
        for pn in pdg_nodes:
            assert pn.prover == Prover.LEAN4
            assert pn.predicate_id in ("sort_step", "search_step")
            assert pn.statement != ""
            assert pn.informal_desc != ""
            assert "concept_type" in pn.context

    def test_uses_type_signature_as_statement(self, atomic_nodes, edges):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        pdg_nodes = to_pdg_nodes(cdg)

        sort_node = next(n for n in pdg_nodes if n.predicate_id == "sort_step")
        assert sort_node.statement == "list[int] -> list[int]"

    def test_rejects_incomplete_cdg(self, non_atomic_nodes):
        cdg = CDGExport(nodes=non_atomic_nodes, edges=[])
        with pytest.raises(HandoffValidationError):
            to_pdg_nodes(cdg)

    def test_context_includes_matched_primitive(self, atomic_nodes, edges):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        pdg_nodes = to_pdg_nodes(cdg)

        sort_node = next(n for n in pdg_nodes if n.predicate_id == "sort_step")
        assert sort_node.context["matched_primitive"] == "heapsort"

    def test_coq_prover(self, atomic_nodes, edges):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        pdg_nodes = to_pdg_nodes(cdg, prover=Prover.COQ)
        for pn in pdg_nodes:
            assert pn.prover == Prover.COQ


class TestSaveLoadJSON:
    def test_roundtrip(self, atomic_nodes, edges, tmp_path):
        cdg = CDGExport(
            nodes=atomic_nodes,
            edges=edges,
            metadata={"goal": "test"},
        )

        path = tmp_path / "cdg.json"
        save_json(cdg, path)
        assert path.exists()

        loaded = load_json(path)
        assert len(loaded.nodes) == len(cdg.nodes)
        assert len(loaded.edges) == len(cdg.edges)
        assert loaded.metadata["goal"] == "test"

    def test_creates_parent_dirs(self, atomic_nodes, edges, tmp_path):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        path = tmp_path / "nested" / "dir" / "cdg.json"
        save_json(cdg, path)
        assert path.exists()

    def test_json_is_valid(self, atomic_nodes, edges, tmp_path):
        cdg = CDGExport(nodes=atomic_nodes, edges=edges)
        path = tmp_path / "cdg.json"
        save_json(cdg, path)

        # Should be valid JSON
        with open(path) as f:
            data = json.load(f)
        assert "nodes" in data
        assert "edges" in data

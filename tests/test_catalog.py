"""Tests for ageom.architect.catalog — PrimitiveCatalog."""

import pytest
from pathlib import Path

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
    NodeStatus,
)


@pytest.fixture
def sample_primitives() -> list[AlgorithmicPrimitive]:
    return [
        AlgorithmicPrimitive(
            name="heapsort",
            source="clrs-30",
            category=ConceptType.SORTING,
            description="Sort using a heap data structure",
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
        ),
        AlgorithmicPrimitive(
            name="dijkstra",
            source="clrs-30",
            category=ConceptType.GRAPH_OPTIMIZATION,
            description="Single-source shortest path in weighted graph",
            inputs=[IOSpec(name="graph", type_desc="Graph")],
            outputs=[IOSpec(name="distances", type_desc="dict[node, float]")],
        ),
        AlgorithmicPrimitive(
            name="binary_search",
            source="clrs-30",
            category=ConceptType.SEARCHING,
            description="Search for element in sorted array using binary division",
            inputs=[IOSpec(name="arr", type_desc="sorted list[int]")],
            outputs=[IOSpec(name="index", type_desc="int")],
        ),
        AlgorithmicPrimitive(
            name="merge_sort",
            source="clrs-30",
            category=ConceptType.SORTING,
            description="Sort by dividing array and merging sorted halves",
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
        ),
    ]


@pytest.fixture
def catalog(sample_primitives) -> PrimitiveCatalog:
    cat = PrimitiveCatalog()
    for p in sample_primitives:
        cat.add(p)
    return cat


class TestPrimitiveCatalogBasics:
    def test_add_and_size(self, catalog):
        assert catalog.size == 4

    def test_get_existing(self, catalog):
        prim = catalog.get("heapsort")
        assert prim is not None
        assert prim.name == "heapsort"

    def test_get_missing(self, catalog):
        assert catalog.get("nonexistent") is None

    def test_search_by_category(self, catalog):
        sorting = catalog.search_by_category(ConceptType.SORTING)
        assert len(sorting) == 2
        names = {p.name for p in sorting}
        assert names == {"heapsort", "merge_sort"}

    def test_search_by_category_empty(self, catalog):
        result = catalog.search_by_category(ConceptType.DYNAMIC_PROGRAMMING)
        assert result == []

    def test_all_primitives(self, catalog):
        all_prims = catalog.all_primitives()
        assert len(all_prims) == 4


class TestIsAtomic:
    def test_matched_by_name(self, catalog):
        node = AlgorithmicNode(
            node_id="n1",
            name="heapsort",
            description="Sort using heap",
            concept_type=ConceptType.SORTING,
            matched_primitive="heapsort",
        )
        assert catalog.is_atomic(node) is True

    def test_matched_by_node_name(self, catalog):
        node = AlgorithmicNode(
            node_id="n1",
            name="binary_search",
            description="Search in sorted array",
            concept_type=ConceptType.SEARCHING,
        )
        assert catalog.is_atomic(node) is True

    def test_not_atomic(self, catalog):
        node = AlgorithmicNode(
            node_id="n1",
            name="Find Optimal Route",
            description="Some complex task",
            concept_type=ConceptType.GRAPH_OPTIMIZATION,
        )
        assert catalog.is_atomic(node) is False


class TestFindMatchingPrimitives:
    def test_finds_category_matches(self, catalog):
        node = AlgorithmicNode(
            node_id="n1",
            name="Sort Items",
            description="Sort a list using a heap data structure",
            concept_type=ConceptType.SORTING,
        )
        matches = catalog.find_matching_primitives(node, k=2)
        assert len(matches) == 2
        # heapsort should score higher due to keyword overlap ("heap", "sort")
        assert matches[0].name == "heapsort"

    def test_cross_category_when_needed(self, catalog):
        node = AlgorithmicNode(
            node_id="n1",
            name="Find shortest path",
            description="Find the shortest path in a weighted graph",
            concept_type=ConceptType.GRAPH_OPTIMIZATION,
        )
        matches = catalog.find_matching_primitives(node, k=5)
        assert len(matches) >= 1
        assert any(m.name == "dijkstra" for m in matches)


class TestSaveLoad:
    def test_roundtrip(self, catalog, tmp_path):
        save_path = tmp_path / "catalog.json"
        catalog.save(save_path)

        loaded = PrimitiveCatalog.load(save_path)
        assert loaded.size == catalog.size

        for prim in catalog.all_primitives():
            loaded_prim = loaded.get(prim.name)
            assert loaded_prim is not None
            assert loaded_prim.name == prim.name
            assert loaded_prim.category == prim.category
            assert loaded_prim.source == prim.source

    def test_save_creates_parent_dirs(self, catalog, tmp_path):
        save_path = tmp_path / "nested" / "dir" / "catalog.json"
        catalog.save(save_path)
        assert save_path.exists()

    def test_load_empty_catalog(self, tmp_path):
        save_path = tmp_path / "empty.json"
        empty = PrimitiveCatalog()
        empty.save(save_path)

        loaded = PrimitiveCatalog.load(save_path)
        assert loaded.size == 0

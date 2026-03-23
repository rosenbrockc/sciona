"""Tests for sciona.architect.catalog — PrimitiveCatalog."""

import pytest

from sciona.architect.catalog import (
    CatalogReport,
    DedupResult,
    PrimitiveCatalog,
    seed_builtin_primitives,
)
from sciona.architect.models import (
    AlgorithmicNode,
    AlgorithmicPrimitive,
    ConceptType,
    IOSpec,
)
from sciona.types import Declaration, Prover


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

    def test_strong_cross_category_match_can_outrank_same_category_prior(self, catalog):
        node = AlgorithmicNode(
            node_id="n1",
            name="Search Sorted Target",
            description="Search the sorted target item using binary division",
            concept_type=ConceptType.SORTING,
            inputs=[
                IOSpec(name="data", type_desc="sorted list[int]"),
                IOSpec(name="target", type_desc="int"),
            ],
            outputs=[IOSpec(name="index", type_desc="int")],
        )

        matches = catalog.find_matching_primitives(node, k=3)
        assert matches[0].name == "binary_search"

    def test_optional_extra_inputs_are_not_filtered_out_of_matches(self):
        catalog = PrimitiveCatalog()
        catalog.add(
            AlgorithmicPrimitive(
                name="filter_required_only",
                source="test",
                category=ConceptType.SIGNAL_FILTER,
                description="Filter a signal with required coefficients and signal",
                inputs=[
                    IOSpec(name="coefficients", type_desc="filter coefficients"),
                    IOSpec(name="signal", type_desc="np.ndarray"),
                ],
                outputs=[IOSpec(name="filtered_signal", type_desc="np.ndarray")],
            )
        )
        catalog.add(
            AlgorithmicPrimitive(
                name="filter_with_optional_mode",
                source="test",
                category=ConceptType.SIGNAL_FILTER,
                description="Filter a signal with optional mode selection",
                inputs=[
                    IOSpec(name="coefficients", type_desc="filter coefficients"),
                    IOSpec(name="signal", type_desc="np.ndarray"),
                    IOSpec(
                        name="mode",
                        type_desc="str",
                        required=False,
                        default_value_repr="'same'",
                    ),
                ],
                outputs=[IOSpec(name="filtered_signal", type_desc="np.ndarray")],
            )
        )

        node = AlgorithmicNode(
            node_id="n1",
            name="Apply Filter",
            description="Filter a signal using coefficients",
            concept_type=ConceptType.SIGNAL_FILTER,
            inputs=[
                IOSpec(name="coefficients", type_desc="filter coefficients"),
                IOSpec(name="signal", type_desc="np.ndarray"),
            ],
            outputs=[IOSpec(name="filtered_signal", type_desc="np.ndarray")],
        )

        matches = catalog.find_matching_primitives(node, k=2)
        assert {match.name for match in matches} == {
            "filter_required_only",
            "filter_with_optional_mode",
        }


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


class TestBuiltinPrimitiveSeeding:
    def test_seed_builtin_primitives_adds_signal_filter_aliases(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        assert catalog.get("parse_filter_spec") is not None
        assert catalog.get("Parse Filter Requirements") is not None
        assert catalog.get("Apply Filter") is not None

    def test_seed_builtin_primitives_adds_signal_transform_aliases(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        assert catalog.get("apply_window_function") is not None
        assert catalog.get("Window") is not None
        assert catalog.get("Forward Transform") is not None
        assert catalog.get("Spectral Processing") is not None
        assert catalog.get("Inverse Transform") is not None

    def test_parse_filter_spec_normalize_design_targets_alias_uses_targets_contract(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        primitive = catalog.get("Normalize Design Targets")

        assert primitive is not None
        assert primitive.name == "parse_filter_spec"
        assert [port.name for port in primitive.inputs] == ["spec"]
        assert [port.name for port in primitive.outputs] == ["design_targets"]

    def test_validate_stability_builtins_cover_polynomial_and_finalize_steps(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        poly = catalog.get("Construct Characteristic Polynomial")
        final = catalog.get("Emit Stable Coefficients")

        assert poly is not None
        assert poly.name == "construct_characteristic_polynomial"
        assert [port.name for port in poly.inputs] == ["normalized_coefficients"]
        assert [port.name for port in poly.outputs] == ["characteristic_polynomial"]

        assert final is not None
        assert final.name == "finalize_stable_coefficients"
        assert [port.name for port in final.inputs] == [
            "normalized_coefficients",
            "stability_report",
        ]
        assert [port.name for port in final.outputs] == ["valid_coefficients"]

    def test_aliases_make_nodes_atomic(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        node = AlgorithmicNode(
            node_id="n1",
            name="Assemble Frequency Response Tuple",
            description="Assemble the final frequency response output",
            concept_type=ConceptType.SIGNAL_FILTER,
        )

        assert catalog.is_atomic(node) is True

    def test_signal_transform_aliases_make_nodes_atomic(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        node = AlgorithmicNode(
            node_id="n2",
            name="Window",
            description="Apply a deterministic window before transformation",
            concept_type=ConceptType.SIGNAL_TRANSFORM,
        )

        assert catalog.is_atomic(node) is True


# ---------------------------------------------------------------------------
# Mock SkillIndex for dedup tests (no torch/FAISS needed)
# ---------------------------------------------------------------------------


class _MockSkillIndex:
    """Minimal mock that returns a canned (Declaration, score) pair."""

    def __init__(self, name: str, score: float):
        self._name = name
        self._score = score

    def search_by_embedding(self, query_text: str, k: int = 1):
        decl = Declaration(
            name=self._name,
            type_signature="",
            prover=Prover.LEAN4,
        )
        return [(decl, self._score)]

    def _primitive_to_text(self, prim):
        return f"{prim.name}: {prim.description}"

    def add_primitive(self, primitive):
        pass


# ---------------------------------------------------------------------------
# Task 1: DedupResult + _structural_match
# ---------------------------------------------------------------------------


class TestStructuralMatch:
    def test_same_category_same_arity(self):
        a = AlgorithmicPrimitive(
            name="a", source="t", category=ConceptType.SORTING,
            description="x",
            inputs=[IOSpec(name="x", type_desc="int")],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        b = AlgorithmicPrimitive(
            name="b", source="t", category=ConceptType.SORTING,
            description="y",
            inputs=[IOSpec(name="x", type_desc="int")],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        assert PrimitiveCatalog._structural_match(a, b) is True

    def test_different_category(self):
        a = AlgorithmicPrimitive(
            name="a", source="t", category=ConceptType.SORTING, description="x",
        )
        b = AlgorithmicPrimitive(
            name="b", source="t", category=ConceptType.SEARCHING, description="y",
        )
        assert PrimitiveCatalog._structural_match(a, b) is False

    def test_arity_off_by_one(self):
        a = AlgorithmicPrimitive(
            name="a", source="t", category=ConceptType.SORTING, description="x",
            inputs=[IOSpec(name="x", type_desc="int")],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        b = AlgorithmicPrimitive(
            name="b", source="t", category=ConceptType.SORTING, description="y",
            inputs=[
                IOSpec(name="x", type_desc="int"),
                IOSpec(name="k", type_desc="int"),
            ],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        assert PrimitiveCatalog._structural_match(a, b) is True

    def test_arity_off_by_two(self):
        a = AlgorithmicPrimitive(
            name="a", source="t", category=ConceptType.SORTING, description="x",
            inputs=[IOSpec(name="x", type_desc="int")],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        b = AlgorithmicPrimitive(
            name="b", source="t", category=ConceptType.SORTING, description="y",
            inputs=[
                IOSpec(name="x", type_desc="int"),
                IOSpec(name="k", type_desc="int"),
                IOSpec(name="n", type_desc="int"),
            ],
            outputs=[IOSpec(name="y", type_desc="int")],
        )
        assert PrimitiveCatalog._structural_match(a, b) is False


# ---------------------------------------------------------------------------
# Task 2: check_duplicate
# ---------------------------------------------------------------------------


class TestCheckDuplicate:
    def test_exact_name_is_duplicate(self, catalog):
        candidate = AlgorithmicPrimitive(
            name="heapsort", source="other", category=ConceptType.SORTING,
            description="Another heapsort",
        )
        result = catalog.check_duplicate(candidate)
        assert result.is_duplicate is True
        assert result.incumbent_name == "heapsort"

    def test_different_name_no_index_not_duplicate(self, catalog):
        candidate = AlgorithmicPrimitive(
            name="quicksort", source="other", category=ConceptType.SORTING,
            description="Sort using pivots",
        )
        result = catalog.check_duplicate(candidate)
        assert result.is_duplicate is False

    def test_mock_high_similarity_same_category(self, catalog):
        mock_idx = _MockSkillIndex("heapsort", 0.92)
        candidate = AlgorithmicPrimitive(
            name="heap_sort_v2", source="other", category=ConceptType.SORTING,
            description="Sort using a heap data structure variant",
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
        )
        result = catalog.check_duplicate(candidate, skill_index=mock_idx)
        assert result.is_duplicate is True
        assert result.incumbent_name == "heapsort"
        assert result.similarity == 0.92

    def test_mock_high_similarity_different_category(self, catalog):
        mock_idx = _MockSkillIndex("heapsort", 0.92)
        candidate = AlgorithmicPrimitive(
            name="heap_priority_queue", source="other",
            category=ConceptType.SEARCHING,  # different category
            description="Priority queue using heap",
        )
        result = catalog.check_duplicate(candidate, skill_index=mock_idx)
        assert result.is_duplicate is False

    def test_mock_below_threshold(self, catalog):
        mock_idx = _MockSkillIndex("heapsort", 0.70)
        candidate = AlgorithmicPrimitive(
            name="timsort", source="other", category=ConceptType.SORTING,
            description="Adaptive merge sort",
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
        )
        result = catalog.check_duplicate(candidate, skill_index=mock_idx)
        assert result.is_duplicate is False


# ---------------------------------------------------------------------------
# Task 3: add_with_dedup
# ---------------------------------------------------------------------------


class TestAddWithDedup:
    def test_no_duplicate_adds_normally(self, catalog):
        candidate = AlgorithmicPrimitive(
            name="quicksort", source="other", category=ConceptType.SORTING,
            description="Sort using pivots",
        )
        result = catalog.add_with_dedup(candidate)
        assert result.is_duplicate is False
        assert catalog.get("quicksort") is not None
        assert catalog.size == 5

    def test_merges_and_keeps_richer(self, catalog):
        # heapsort is already in catalog with a short description
        mock_idx = _MockSkillIndex("heapsort", 0.95)
        candidate = AlgorithmicPrimitive(
            name="heap_sort_improved", source="other",
            category=ConceptType.SORTING,
            description="Improved heapsort with optimised sift-down "
                        "and better cache locality for large arrays",
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
            type_signature="list[int] -> list[int]",
        )
        result = catalog.add_with_dedup(candidate, skill_index=mock_idx)
        assert result.is_duplicate is True
        # Candidate has richer metadata (longer description + type_signature)
        winner = catalog.get("heap_sort_improved")
        assert winner is not None
        assert "Improved" in winner.description
        # Old name should resolve via alias
        assert catalog.get("heapsort") is not None

    def test_merges_and_keeps_incumbent(self, catalog):
        mock_idx = _MockSkillIndex("heapsort", 0.90)
        candidate = AlgorithmicPrimitive(
            name="hs2", source="other", category=ConceptType.SORTING,
            description="hs",  # very short — incumbent is richer
            inputs=[IOSpec(name="arr", type_desc="list[int]")],
            outputs=[IOSpec(name="sorted", type_desc="list[int]")],
        )
        result = catalog.add_with_dedup(candidate, skill_index=mock_idx)
        assert result.is_duplicate is True
        # Incumbent "heapsort" wins
        assert catalog.get("heapsort") is not None
        assert catalog.get("heapsort").name == "heapsort"
        # Candidate name resolves as alias
        assert catalog.get("hs2") is not None
        assert catalog.get("hs2").name == "heapsort"

    def test_no_index_falls_through(self, catalog):
        candidate = AlgorithmicPrimitive(
            name="introsort", source="other", category=ConceptType.SORTING,
            description="Hybrid sorting algorithm",
        )
        result = catalog.add_with_dedup(candidate, skill_index=None)
        assert result.is_duplicate is False
        assert catalog.get("introsort") is not None


# ---------------------------------------------------------------------------
# Task 6: CatalogReport
# ---------------------------------------------------------------------------


class TestCatalogReport:
    def test_counts(self, catalog):
        report = CatalogReport()
        mock_idx = _MockSkillIndex("heapsort", 0.95)

        # Non-duplicate
        catalog.add_with_dedup(
            AlgorithmicPrimitive(
                name="quicksort", source="t", category=ConceptType.SORTING,
                description="Pivot sort",
            ),
            report=report,
        )
        # Duplicate
        catalog.add_with_dedup(
            AlgorithmicPrimitive(
                name="heap_sort_v2", source="t", category=ConceptType.SORTING,
                description="Another heap sort",
                inputs=[IOSpec(name="arr", type_desc="list[int]")],
                outputs=[IOSpec(name="sorted", type_desc="list[int]")],
            ),
            skill_index=mock_idx,
            report=report,
        )

        assert report.total_candidates == 2
        assert report.added == 1
        assert report.merged == 1
        assert len(report.merge_details) == 1
        assert report.merge_details[0][0] == "heap_sort_v2"
        assert report.merge_details[0][1] == "heapsort"


# ---------------------------------------------------------------------------
# Task 7: find_gaps
# ---------------------------------------------------------------------------


class TestFindGaps:
    def test_returns_clusters(self, catalog):
        nodes = [
            AlgorithmicNode(
                node_id="a", name="Parse Requirements",
                description="Parse and normalize input requirements into typed constraints",
                concept_type=ConceptType.CUSTOM,
            ),
            AlgorithmicNode(
                node_id="b", name="Normalize Requirements",
                description="Normalize input requirements into typed constraints",
                concept_type=ConceptType.CUSTOM,
            ),
            AlgorithmicNode(
                node_id="c", name="Compute FFT",
                description="Compute fast Fourier transform of a signal",
                concept_type=ConceptType.SIGNAL_TRANSFORM,
            ),
        ]
        clusters = catalog.find_gaps(nodes)
        # a and b overlap ("normalize", "input", "requirements", "typed", "constraints")
        # c is alone -> not in a cluster
        assert len(clusters) >= 1
        cluster_names = [{n.node_id for n in c} for c in clusters]
        assert {"a", "b"} in cluster_names

    def test_no_gaps_when_all_matched(self, catalog):
        nodes = [
            AlgorithmicNode(
                node_id="a", name="heapsort",
                description="Sort using heap",
                concept_type=ConceptType.SORTING,
            ),
        ]
        clusters = catalog.find_gaps(nodes)
        assert clusters == []

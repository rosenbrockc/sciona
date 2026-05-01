"""Tests for the high-level sciona.api module."""

from __future__ import annotations

from pathlib import Path

import pytest

from sciona.sdk import Sciona, load_catalog_from_repos, _resolve_default_repos
from sciona.api_models import (
    AtomMatch,
    AtomSearchResult,
    CDGInspection,
    GapReport,
    GroundingReport,
    ProposalResult,
    StageMatchResult,
)
from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from sciona.architect.models import AlgorithmicPrimitive, ConceptType, IOSpec

PERSONAL = Path.home() / "personal"

ATOM_REPOS = [
    PERSONAL / repo
    for repo in [
        "sciona-atoms",
        "sciona-atoms-ml",
        "sciona-atoms-dl",
        "sciona-atoms-cs",
        "sciona-atoms-bio",
        "sciona-atoms-physics",
        "sciona-atoms-signal",
        "sciona-atoms-geo",
        "sciona-atoms-fintech",
        "sciona-atoms-robotics",
    ]
    if (PERSONAL / repo).exists()
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def full_catalog() -> PrimitiveCatalog:
    return load_catalog_from_repos(ATOM_REPOS)


@pytest.fixture(scope="module")
def sciona_instance(full_catalog: PrimitiveCatalog) -> Sciona:
    return Sciona.from_catalog(full_catalog)


# ---------------------------------------------------------------------------
# Catalog loading
# ---------------------------------------------------------------------------


class TestCatalogLoading:
    def test_load_catalog_from_repos_returns_catalog(self, full_catalog):
        assert isinstance(full_catalog, PrimitiveCatalog)
        assert full_catalog.size > 1000

    def test_load_catalog_includes_aliases(self, full_catalog):
        # Aliases declared in cdg.json should be loadable via get()
        prim = full_catalog.get("collaborative_filtering")
        assert prim is not None
        assert prim.name == "als_user_update"

    def test_load_catalog_includes_new_atoms(self, full_catalog):
        # Atoms from the research round should be present
        for name in [
            "elo_rating_update",
            "levenshtein_distance",
            "wavelet_denoise",
        ]:
            assert full_catalog.get(name) is not None, f"{name} not found"

    def test_resolve_default_repos_finds_repos(self):
        repos = _resolve_default_repos()
        assert len(repos) >= 5


# ---------------------------------------------------------------------------
# Sciona construction
# ---------------------------------------------------------------------------


class TestScionaConstruction:
    def test_from_catalog(self, full_catalog):
        s = Sciona.from_catalog(full_catalog)
        assert s.atom_count == full_catalog.size
        assert s.catalog is full_catalog

    def test_repr(self, sciona_instance):
        r = repr(sciona_instance)
        assert "Sciona" in r
        assert "atoms=" in r


# ---------------------------------------------------------------------------
# Search & retrieval
# ---------------------------------------------------------------------------


class TestSearchAndRetrieval:
    def test_search_atoms_returns_results(self, sciona_instance):
        results = sciona_instance.search_atoms("kalman filter", k=5)
        assert len(results) == 5
        assert all(isinstance(r, AtomSearchResult) for r in results)

    def test_search_atoms_finds_relevant(self, sciona_instance):
        results = sciona_instance.search_atoms("ensemble voting", k=5)
        names = [r.atom_name for r in results]
        assert any("voting" in n for n in names)

    def test_find_matching_atoms(self, sciona_instance):
        results = sciona_instance.find_matching_atoms(
            "compute TF-IDF features from text documents",
            concept_type="data_extraction",
            k=5,
        )
        assert len(results) == 5
        assert all(isinstance(r, AtomMatch) for r in results)
        names = [r.atom_name for r in results]
        assert any("tfidf" in n for n in names)

    def test_find_matching_atoms_alias_boost(self, sciona_instance):
        """Atoms with matching aliases should rank high."""
        results = sciona_instance.find_matching_atoms(
            "collaborative filtering using ALS",
            k=3,
        )
        names = [r.atom_name for r in results]
        assert "als_user_update" in names


# ---------------------------------------------------------------------------
# Grounding report
# ---------------------------------------------------------------------------


class TestGroundingReport:
    def test_grounding_rate_empty(self):
        g = GroundingReport()
        assert g.grounding_rate == 0.0

    def test_grounding_rate_full(self):
        g = GroundingReport(total_stages=10, bound_active=10)
        assert g.grounding_rate == 1.0

    def test_grounding_rate_mixed(self):
        g = GroundingReport(
            total_stages=10,
            bound_active=3,
            orchestration=2,
            trivial_inline=1,
            external_knowledge=1,
            unbound=3,
        )
        assert g.grounding_rate == pytest.approx(0.7)

    def test_build_grounding_report(self, sciona_instance):
        matches = [
            StageMatchResult("a", "A", "replace_stage", matched_atom="x"),
            StageMatchResult("b", "B", "orchestration"),
            StageMatchResult("c", "C", "replace_stage"),  # unbound
        ]
        report = Sciona._build_grounding_report(matches)
        assert report.total_stages == 3
        assert report.bound_active == 1
        assert report.orchestration == 1
        assert report.unbound == 1


# ---------------------------------------------------------------------------
# CDG inspection
# ---------------------------------------------------------------------------


class TestInspection:
    def test_inspect_cdg_smoke(self, sciona_instance):
        """Load a real CDG and inspect it."""
        cdg_path = (
            PERSONAL
            / "sciona-atoms"
            / "data"
            / "solution_cdgs"
            / "adversarial_attacks_1st.json"
        )
        if not cdg_path.exists():
            pytest.skip("CDG file not found")

        import json
        from sciona.architect.handoff import CDGExport
        from sciona.architect.models import AlgorithmicNode, DependencyEdge

        data = json.loads(cdg_path.read_text())

        # Build minimal CDGExport-compatible object
        class _CDG:
            def __init__(self, data):
                self.nodes = [
                    AlgorithmicNode(
                        node_id=s["stage_id"],
                        name=s.get("name", s["stage_id"]),
                        description=s.get("description", ""),
                        concept_type=ConceptType(
                            s.get("concept_type", "custom")
                        ),
                        inputs=[
                            IOSpec(**inp) for inp in s.get("inputs", [])
                        ],
                        outputs=[
                            IOSpec(**out) for out in s.get("outputs", [])
                        ],
                    )
                    for s in data.get("stages", [])
                ]
                self.edges = []

        cdg = _CDG(data)
        inspection = sciona_instance.inspect_cdg(cdg)
        assert isinstance(inspection, CDGInspection)
        assert inspection.total_stages > 0
        assert isinstance(inspection.grounding, GroundingReport)

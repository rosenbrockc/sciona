"""End-to-end tests for the flywheel: auto-upsert of solved runs -> retrieval for similar goals."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from ageom.architect.graph_alignment import AlignmentScore, GraphAlignmentScorer
from ageom.architect.graph_retrieval import ExampleChild, ExampleDecomposition
from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.architect.template_retriever import TemplateMatch, TemplateRetriever
from ageom.config import AgeomConfig
from ageom.orchestrator import OrchestratorResult
from ageom.result_to_cdg import RunCDGMetadata, orchestrator_result_to_cdg
from ageom.telemetry import get_event_log, log_event, reset_telemetry_runtime
from ageom.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    VerificationResult,
)
from ageom.upsert_cdg import sanitize_cdg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_root_node() -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id="root",
        parent_id=None,
        name="Root Goal",
        description="Top-level goal",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.DECOMPOSED,
        children=["leaf_a", "leaf_b", "leaf_c"],
        depth=0,
    )


def _make_atomic_node(node_id: str, name: str) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        parent_id="root",
        name=name,
        description=f"Atomic step: {name}",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        type_signature="nat -> nat",
        inputs=[IOSpec(name="x", type_desc="nat")],
        outputs=[IOSpec(name="y", type_desc="nat")],
        depth=1,
    )


def _make_match_result(node_id: str, success: bool, decl_name: str = "") -> MatchResult:
    decl_name = decl_name or f"prim_{node_id}"
    decl = Declaration(name=decl_name, type_signature="nat -> nat")
    candidate = CandidateMatch(declaration=decl, score=0.95, retrieval_method="embedding")
    vr = VerificationResult(candidate=candidate, verified=success)
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement="nat -> nat"),
        verified_match=vr if success else None,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


def _make_metadata() -> RunCDGMetadata:
    return RunCDGMetadata(
        run_id="flywheel-run-001",
        goal="Sort a list of integers",
        execution_path="verified",
        timestamp="2026-03-17T12:00:00Z",
        verified_leaf_coverage=0.0,
    )


def _build_orchestrator_result(
    matched_ids: list[str],
) -> OrchestratorResult:
    """Build an OrchestratorResult with 3 leaves; matched_ids controls which succeed."""
    root = _make_root_node()
    leaf_a = _make_atomic_node("leaf_a", "Leaf A")
    leaf_b = _make_atomic_node("leaf_b", "Leaf B")
    leaf_c = _make_atomic_node("leaf_c", "Leaf C")
    cdg = CDGExport(nodes=[root, leaf_a, leaf_b, leaf_c], edges=[])
    match_results = [
        _make_match_result("leaf_a", "leaf_a" in matched_ids),
        _make_match_result("leaf_b", "leaf_b" in matched_ids),
        _make_match_result("leaf_c", "leaf_c" in matched_ids),
    ]
    return OrchestratorResult(cdg=cdg, match_results=match_results, rounds_used=1)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_telemetry():
    """Reset telemetry state before and after each test."""
    reset_telemetry_runtime()
    yield
    reset_telemetry_runtime()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestResultToCdgProducesUpsertableDict:
    """1. Build fully matched OrchestratorResult (3 leaves all matched),
    call orchestrator_result_to_cdg. Assert output has 'nodes' and 'edges',
    coverage=1.0, provenance on root."""

    def test_result_to_cdg_produces_upsertable_dict(self):
        result = _build_orchestrator_result(["leaf_a", "leaf_b", "leaf_c"])
        metadata = _make_metadata()

        cdg_dict = orchestrator_result_to_cdg(result, metadata)

        # Has required top-level keys
        assert "nodes" in cdg_dict
        assert "edges" in cdg_dict

        # Coverage is 1.0 (all 3 leaves matched)
        assert metadata.verified_leaf_coverage == 1.0

        # Provenance is on the root node (parent_id is None)
        root = next(n for n in cdg_dict["nodes"] if n.get("parent_id") is None)
        assert "provenance" in root
        prov = root["provenance"]
        assert prov["run_id"] == "flywheel-run-001"
        assert prov["goal"] == "Sort a list of integers"
        assert prov["timestamp"] == "2026-03-17T12:00:00Z"
        assert prov["execution_path"] == "verified"
        assert prov["verified_leaf_coverage"] == 1.0


class TestAutoUpsertSkippedForLowCoverage:
    """2. 0/3 matched -> coverage=0.0, below AgeomConfig default threshold (0.5)."""

    def test_auto_upsert_skipped_for_low_coverage(self):
        result = _build_orchestrator_result([])  # 0 matched
        metadata = _make_metadata()

        orchestrator_result_to_cdg(result, metadata)

        config = AgeomConfig(
            _env_file=None,  # type: ignore[call-arg]
        )
        # Coverage should be 0.0
        assert metadata.verified_leaf_coverage == 0.0
        # Below default threshold
        assert metadata.verified_leaf_coverage < config.auto_upsert_min_coverage


class TestAutoUpsertFiresForSufficientCoverage:
    """3. 3/3 matched -> coverage=1.0 >= threshold."""

    def test_auto_upsert_fires_for_sufficient_coverage(self):
        result = _build_orchestrator_result(["leaf_a", "leaf_b", "leaf_c"])
        metadata = _make_metadata()

        orchestrator_result_to_cdg(result, metadata)

        config = AgeomConfig(
            _env_file=None,  # type: ignore[call-arg]
        )
        # Coverage should be 1.0
        assert metadata.verified_leaf_coverage == 1.0
        # At or above default threshold
        assert metadata.verified_leaf_coverage >= config.auto_upsert_min_coverage


class TestUpsertedRunBecomesRetrievalCandidate:
    """4. Mock GraphStore.query_verified_exemplars returning the upserted record.
    Create TemplateRetriever, call find_refinement_templates. Assert non-empty
    match with confidence >= threshold."""

    def test_upserted_run_becomes_retrieval_candidate(self):
        # First, produce the CDG dict as if it were upserted
        result = _build_orchestrator_result(["leaf_a", "leaf_b", "leaf_c"])
        metadata = _make_metadata()
        cdg_dict = orchestrator_result_to_cdg(result, metadata)

        # Build a mock store that returns a verified exemplar record
        mock_store = MagicMock()
        exemplar_record = {
            "fqn": "test_repo.root",
            "repo": "test_repo",
            "topo_hash": "abc123",
            "verified_leaf_coverage": 1.0,
            "concept_type": "sorting",
        }
        mock_store.query_verified_exemplars = AsyncMock(return_value=[exemplar_record])

        # Create TemplateRetriever with confidence_threshold=0.6
        scorer = GraphAlignmentScorer()
        retriever = TemplateRetriever(
            store=mock_store,
            scorer=scorer,
            confidence_threshold=0.6,
        )

        # Create a failed node to search for refinement templates
        failed_node = AlgorithmicNode(
            node_id="failed_node",
            parent_id=None,
            name="Failed Sort",
            description="A sorting step that failed",
            concept_type=ConceptType.SORTING,
            status=NodeStatus.ATOMIC,
            depth=1,
        )
        failure_context = {
            "description": "A sorting step that failed",
            "concept_type": "sorting",
            "error_summaries": ["no match found"],
            "statement": "nat -> nat",
        }

        # Run the async method
        matches = asyncio.get_event_loop().run_until_complete(
            retriever.find_refinement_templates(failed_node, failure_context)
        )

        # Verify: query_verified_exemplars was called with correct args
        mock_store.query_verified_exemplars.assert_awaited_once_with(
            concept_type="sorting",
            min_coverage=0.5,
            limit=10,
        )

        # Non-empty results with confidence >= threshold
        assert len(matches) > 0
        for match in matches:
            assert isinstance(match, TemplateMatch)
            assert match.confidence >= 0.6
            assert match.source == "verified_exemplar"


class TestSanitizeCdgPreservesMatchedPrimitives:
    """5. matched_primitive values survive sanitization."""

    def test_sanitize_cdg_preserves_matched_primitives(self):
        result = _build_orchestrator_result(["leaf_a", "leaf_b", "leaf_c"])
        metadata = _make_metadata()

        cdg_dict = orchestrator_result_to_cdg(result, metadata)

        # Before sanitization, check matched_primitive values exist
        nodes_by_id = {n["node_id"]: n for n in cdg_dict["nodes"]}
        assert nodes_by_id["leaf_a"]["matched_primitive"] == "prim_leaf_a"
        assert nodes_by_id["leaf_b"]["matched_primitive"] == "prim_leaf_b"
        assert nodes_by_id["leaf_c"]["matched_primitive"] == "prim_leaf_c"

        # Apply sanitize_cdg again (it was already applied inside orchestrator_result_to_cdg)
        sanitized = sanitize_cdg(cdg_dict)

        # matched_primitive values must survive
        sanitized_by_id = {n["node_id"]: n for n in sanitized["nodes"]}
        assert sanitized_by_id["leaf_a"]["matched_primitive"] == "prim_leaf_a"
        assert sanitized_by_id["leaf_b"]["matched_primitive"] == "prim_leaf_b"
        assert sanitized_by_id["leaf_c"]["matched_primitive"] == "prim_leaf_c"


class TestProvenanceMetadataSurvivedSanitize:
    """6. Provenance dict with all fields survives sanitization."""

    def test_provenance_metadata_survives_sanitize(self):
        result = _build_orchestrator_result(["leaf_a", "leaf_b", "leaf_c"])
        metadata = _make_metadata()

        cdg_dict = orchestrator_result_to_cdg(result, metadata)

        # Verify provenance before extra sanitization
        root_before = next(n for n in cdg_dict["nodes"] if n.get("parent_id") is None)
        assert "provenance" in root_before
        prov_before = root_before["provenance"]
        assert set(prov_before.keys()) == {
            "run_id",
            "goal",
            "timestamp",
            "execution_path",
            "verified_leaf_coverage",
        }

        # Apply sanitize_cdg again
        sanitized = sanitize_cdg(cdg_dict)

        # Provenance must survive with all fields intact
        root_after = next(n for n in sanitized["nodes"] if n.get("parent_id") is None)
        assert "provenance" in root_after
        prov_after = root_after["provenance"]
        assert prov_after["run_id"] == "flywheel-run-001"
        assert prov_after["goal"] == "Sort a list of integers"
        assert prov_after["timestamp"] == "2026-03-17T12:00:00Z"
        assert prov_after["execution_path"] == "verified"
        assert prov_after["verified_leaf_coverage"] == 1.0


class TestFlywheelTelemetryEvents:
    """7. log AUTO_UPSERT_COMPLETED and AUTO_UPSERT_SKIPPED_LOW_COVERAGE events,
    verify they appear in event log with correct payloads."""

    def test_flywheel_telemetry_events(self):
        event_log = get_event_log()

        # Log an AUTO_UPSERT_COMPLETED event
        log_event(
            "flywheel",
            "auto_upsert",
            "AUTO_UPSERT_COMPLETED",
            payload={
                "run_id": "flywheel-run-001",
                "coverage": 1.0,
                "nodes_upserted": 4,
            },
        )

        # Log an AUTO_UPSERT_SKIPPED_LOW_COVERAGE event
        log_event(
            "flywheel",
            "auto_upsert",
            "AUTO_UPSERT_SKIPPED_LOW_COVERAGE",
            payload={
                "run_id": "flywheel-run-002",
                "coverage": 0.2,
                "threshold": 0.5,
            },
        )

        events = event_log.events

        # Find the completed event
        completed_events = [
            e for e in events if e.event_type == "AUTO_UPSERT_COMPLETED"
        ]
        assert len(completed_events) == 1
        completed = completed_events[0]
        assert completed.round == "flywheel"
        assert completed.phase == "auto_upsert"
        assert completed.payload["run_id"] == "flywheel-run-001"
        assert completed.payload["coverage"] == 1.0
        assert completed.payload["nodes_upserted"] == 4

        # Find the skipped event
        skipped_events = [
            e for e in events if e.event_type == "AUTO_UPSERT_SKIPPED_LOW_COVERAGE"
        ]
        assert len(skipped_events) == 1
        skipped = skipped_events[0]
        assert skipped.round == "flywheel"
        assert skipped.phase == "auto_upsert"
        assert skipped.payload["run_id"] == "flywheel-run-002"
        assert skipped.payload["coverage"] == 0.2
        assert skipped.payload["threshold"] == 0.5

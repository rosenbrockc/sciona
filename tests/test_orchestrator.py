"""Tests for the orchestrator feedback loop (Issue 2)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.orchestrator import OrchestratorResult, refine_on_failure, run_orchestration
from ageom.types import (
    CandidateMatch,
    Declaration,
    FailureAction,
    MatchFailureReport,
    MatchResult,
    PDGNode,
    Prover,
    VerificationResult,
)


def _make_atomic_node(node_id: str, name: str) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=f"Test {name}",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        type_signature="nat -> nat",
        inputs=[IOSpec(name="x", type_desc="nat")],
        outputs=[IOSpec(name="y", type_desc="nat")],
    )


def _make_cdg(*node_ids: str) -> CDGExport:
    nodes = [_make_atomic_node(nid, f"node_{nid}") for nid in node_ids]
    return CDGExport(nodes=nodes, edges=[])


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


@pytest.mark.asyncio
async def test_orchestration_all_matched():
    """When all nodes match, orchestration completes in one round."""
    cdg = _make_cdg("a", "b")

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(side_effect=[
        _make_match_result("a", True),
        _make_match_result("b", True),
    ])

    llm = AsyncMock()

    result = await run_orchestration(
        cdg, hunter_agent=hunter, llm=llm, max_rounds=3,
    )

    assert result.all_matched
    assert result.rounds_used == 1
    assert len(result.match_results) == 2


@pytest.mark.asyncio
async def test_orchestration_failure_triggers_refinement():
    """When a node fails, the orchestrator should refine and retry."""
    cdg = _make_cdg("a")

    # First call fails, second succeeds (after split creates sub-nodes)
    call_count = 0

    async def mock_find_match(pdg_node):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return _make_match_result(pdg_node.predicate_id, False)
        return _make_match_result(pdg_node.predicate_id, True)

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(side_effect=mock_find_match)

    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='[{"name": "sub1", "description": "sub", "type_signature": "nat -> nat"}]')

    result = await run_orchestration(
        cdg, hunter_agent=hunter, llm=llm, max_rounds=3,
    )

    assert result.rounds_used >= 1
    assert len(result.failures) >= 1


def test_match_failure_report_from_match_result():
    """MatchFailureReport.from_match_result creates correct report."""
    mr = _make_match_result("x", False)
    report = MatchFailureReport.from_match_result(mr)
    assert report.pdg_node.predicate_id == "x"
    assert len(report.best_candidates) <= 5


@pytest.mark.asyncio
async def test_refine_on_failure_ungroundable():
    """UNGROUNDABLE action marks the node as rejected."""
    cdg = _make_cdg("a")
    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="a", statement="nat -> nat"),
        suggested_action=FailureAction.UNGROUNDABLE,
    )
    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm)
    rejected = [n for n in updated.nodes if n.status == NodeStatus.REJECTED]
    assert len(rejected) == 1

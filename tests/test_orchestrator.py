"""Tests for the orchestrator feedback loop (Issue 2)."""

import asyncio
from unittest.mock import AsyncMock

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    IOSpec,
    NodeStatus,
)
from sciona.orchestrator import refine_on_failure, run_orchestration
from sciona.types import (
    CandidateMatch,
    Declaration,
    FailureAction,
    MatchFailureReport,
    MatchResult,
    PDGNode,
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
    candidate = CandidateMatch(
        declaration=decl, score=0.9, retrieval_method="embedding"
    )
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
    hunter.find_match = AsyncMock(
        side_effect=[
            _make_match_result("a", True),
            _make_match_result("b", True),
        ]
    )

    llm = AsyncMock()

    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=llm,
        max_rounds=3,
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
    llm.complete = AsyncMock(
        return_value='[{"name": "sub1", "description": "sub", "type_signature": "nat -> nat"}]'
    )

    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=llm,
        max_rounds=3,
    )

    assert result.rounds_used >= 1
    assert len(result.failures) >= 1


@pytest.mark.asyncio
async def test_orchestration_runs_hunter_in_parallel_when_enabled():
    """Hunter calls should overlap when concurrency > 1."""
    cdg = _make_cdg("a", "b", "c")

    active = 0
    max_active = 0

    async def mock_find_match(pdg_node):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _make_match_result(pdg_node.predicate_id, True)

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(side_effect=mock_find_match)
    llm = AsyncMock()

    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=llm,
        max_rounds=1,
        hunter_concurrency=3,
    )

    assert result.all_matched
    assert max_active > 1


@pytest.mark.asyncio
async def test_orchestration_stops_before_hunter_on_blocked_cdg():
    blocked = AlgorithmicNode(
        node_id="blocked",
        name="Blocked Step",
        description="blocked",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.BLOCKED,
        depth=0,
    )
    cdg = CDGExport(
        nodes=[blocked],
        edges=[],
        metadata={"architect_error": "decomposition blocked"},
    )

    hunter = AsyncMock()
    llm = AsyncMock()

    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=llm,
        max_rounds=3,
    )

    assert result.rounds_used == 0
    assert result.match_results == []
    hunter.find_match.assert_not_called()


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


@pytest.mark.asyncio
async def test_refine_on_failure_uses_deterministic_filter_split_before_llm():
    node = AlgorithmicNode(
        node_id="filter_step",
        name="Bandpass ECG Filter",
        description="Design and apply a stable bandpass filter to ECG samples.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        depth=0,
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="filter_step",
            statement="Bandpass raw ECG into cardiac frequency region",
            informal_desc="stable digital filter design and application",
        ),
        error_summaries=["Expected filtered_signal but got response tuple"],
        suggested_action=FailureAction.SPLIT,
    )
    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm)

    parent = next(n for n in updated.nodes if n.node_id == "filter_step")
    children = [n for n in updated.nodes if n.parent_id == "filter_step"]
    assert parent.status == NodeStatus.DECOMPOSED
    assert [child.name for child in children] == ["Design Filter", "Apply Filter"]
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_refine_generalize_broadens_description():
    """GENERALIZE strips algorithm-specific names from description."""
    node = AlgorithmicNode(
        node_id="solve_step",
        name="Dijkstra Solver",
        description="Use Dijkstra's algorithm to find shortest paths in weighted graph",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        type_signature="Graph -> Distances",
        matched_primitive="dijkstra_shortest_path",
        primitive_binding_confidence=0.5,
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="solve_step", statement="shortest path"),
        error_summaries=["type mismatch"],
        suggested_action=FailureAction.GENERALIZE,
    )
    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm)
    n = next(n for n in updated.nodes if n.node_id == "solve_step")
    assert "Dijkstra" not in n.description
    assert "shortest path" in n.description.lower() or "weighted graph" in n.description.lower()


@pytest.mark.asyncio
async def test_refine_generalize_clears_type_signature():
    """GENERALIZE clears type_signature."""
    node = AlgorithmicNode(
        node_id="step_a",
        name="Solver",
        description="Solve using Cholesky factorization for SPD system",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        type_signature="Matrix -> Vector",
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="step_a", statement="linear solve"),
        suggested_action=FailureAction.GENERALIZE,
    )
    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm)
    n = next(n for n in updated.nodes if n.node_id == "step_a")
    assert n.type_signature == ""


@pytest.mark.asyncio
async def test_refine_generalize_resets_primitive_binding():
    """GENERALIZE resets matched_primitive and primitive_binding_confidence."""
    node = AlgorithmicNode(
        node_id="step_b",
        name="Filter",
        description="Apply Butterworth bandpass filter to signal",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        matched_primitive="butterworth_bandpass",
        primitive_binding_confidence=0.8,
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="step_b", statement="bandpass filter"),
        suggested_action=FailureAction.GENERALIZE,
    )
    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm)
    n = next(n for n in updated.nodes if n.node_id == "step_b")
    assert n.matched_primitive is None
    assert n.primitive_binding_confidence == 0.0
    assert "GENERALIZE" in n.critic_notes


@pytest.mark.asyncio
async def test_refine_on_failure_falls_back_to_llm_for_generic_split():
    cdg = _make_cdg("a")
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="a",
            statement="nat -> nat",
            informal_desc="generic helper",
        ),
        error_summaries=["type mismatch"],
        suggested_action=FailureAction.SPLIT,
    )
    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value='[{"name": "sub1", "description": "sub", "type_signature": "nat -> nat"}]'
    )

    updated = await refine_on_failure(failure, cdg, llm)

    parent = next(n for n in updated.nodes if n.node_id == "a")
    children = [n for n in updated.nodes if n.parent_id == "a"]
    assert parent.status == NodeStatus.DECOMPOSED
    assert len(children) == 1
    llm.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# Retrieval-based refinement tests
# ---------------------------------------------------------------------------


def _make_template_match(children, confidence=0.85):
    """Create a mock TemplateMatch with the given children and confidence."""
    from dataclasses import dataclass

    @dataclass
    class _FakeChild:
        name: str
        description: str
        type_signature: str

    @dataclass
    class _FakeExample:
        children: list

    @dataclass
    class _FakeMatch:
        example: _FakeExample
        confidence: float
        source: str

    fake_children = [
        _FakeChild(name=c["name"], description=c["description"], type_signature=c.get("type_signature", ""))
        for c in children
    ]
    return _FakeMatch(
        example=_FakeExample(children=fake_children),
        confidence=confidence,
        source="verified_exemplar",
    )


@pytest.mark.asyncio
async def test_refine_split_uses_retrieval():
    """When template_retriever returns a good match (>=0.7), use retrieval-based split."""
    node = AlgorithmicNode(
        node_id="retrieve_step",
        name="Matrix Decompose",
        description="Decompose matrix into lower-upper factors",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        depth=0,
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="retrieve_step",
            statement="LU decomposition of matrix",
            informal_desc="decompose matrix into lower-upper factors",
        ),
        error_summaries=["no matching primitive found"],
        suggested_action=FailureAction.SPLIT,
    )

    retrieved_children = [
        {"name": "Factor Lower", "description": "compute lower triangular factor"},
        {"name": "Factor Upper", "description": "compute upper triangular factor"},
    ]
    template_retriever = AsyncMock()
    template_retriever.find_refinement_templates = AsyncMock(
        return_value=[_make_template_match(retrieved_children, confidence=0.85)]
    )

    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm, template_retriever=template_retriever)

    parent = next(n for n in updated.nodes if n.node_id == "retrieve_step")
    children = [n for n in updated.nodes if n.parent_id == "retrieve_step"]
    assert parent.status == NodeStatus.DECOMPOSED
    assert len(children) == 2
    assert children[0].name == "Factor Lower"
    assert children[1].name == "Factor Upper"
    template_retriever.find_refinement_templates.assert_awaited_once()
    # LLM should NOT have been called
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_refine_split_falls_back_to_deterministic():
    """When template_retriever returns empty, fall back to deterministic patterns."""
    node = AlgorithmicNode(
        node_id="filter_step",
        name="Bandpass ECG Filter",
        description="Design and apply a stable bandpass filter to ECG samples.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        depth=0,
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="filter_step",
            statement="Bandpass raw ECG into cardiac frequency region",
            informal_desc="stable digital filter design and application",
        ),
        error_summaries=["Expected filtered_signal but got response tuple"],
        suggested_action=FailureAction.SPLIT,
    )

    template_retriever = AsyncMock()
    template_retriever.find_refinement_templates = AsyncMock(return_value=[])

    llm = AsyncMock()

    updated = await refine_on_failure(failure, cdg, llm, template_retriever=template_retriever)

    parent = next(n for n in updated.nodes if n.node_id == "filter_step")
    children = [n for n in updated.nodes if n.parent_id == "filter_step"]
    assert parent.status == NodeStatus.DECOMPOSED
    # Deterministic filter split produces Design Filter + Apply Filter
    assert [child.name for child in children] == ["Design Filter", "Apply Filter"]
    template_retriever.find_refinement_templates.assert_awaited_once()
    llm.complete.assert_not_called()


@pytest.mark.asyncio
async def test_refine_split_falls_back_to_llm():
    """When both retrieval and deterministic return nothing, fall back to LLM."""
    cdg = _make_cdg("a")
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="a",
            statement="nat -> nat",
            informal_desc="generic helper",
        ),
        error_summaries=["type mismatch"],
        suggested_action=FailureAction.SPLIT,
    )

    template_retriever = AsyncMock()
    template_retriever.find_refinement_templates = AsyncMock(return_value=[])

    llm = AsyncMock()
    llm.complete = AsyncMock(
        return_value='[{"name": "sub1", "description": "sub", "type_signature": "nat -> nat"}]'
    )

    updated = await refine_on_failure(failure, cdg, llm, template_retriever=template_retriever)

    parent = next(n for n in updated.nodes if n.node_id == "a")
    children = [n for n in updated.nodes if n.parent_id == "a"]
    assert parent.status == NodeStatus.DECOMPOSED
    assert len(children) == 1
    template_retriever.find_refinement_templates.assert_awaited_once()
    llm.complete.assert_awaited_once()

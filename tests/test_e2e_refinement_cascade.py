"""End-to-end tests for the orchestrator 3-layer refinement cascade.

Tests cover:
  1. Retrieval hit skips deterministic and LLM layers
  2. Deterministic fallback when retrieval confidence is below threshold
  3. LLM fallback when no deterministic match
  4. UNGROUNDABLE action marks node rejected
  5. GENERALIZE action strips algorithm names
  6. Multi-round orchestration with split + retry
  7. Max rounds reached populates ungroundable list
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    IOSpec,
    NodeStatus,
)
from ageom.orchestrator import refine_on_failure, run_orchestration
from ageom.telemetry import get_event_log, reset_telemetry_runtime
from ageom.types import (
    CandidateMatch,
    Declaration,
    FailureAction,
    MatchFailureReport,
    MatchResult,
    PDGNode,
    VerificationResult,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clean_telemetry():
    """Reset global telemetry state before each test."""
    reset_telemetry_runtime()
    yield
    reset_telemetry_runtime()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_atomic_node(
    node_id: str,
    name: str,
    *,
    description: str = "",
    concept_type: ConceptType = ConceptType.SORTING,
    type_signature: str = "nat -> nat",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=description or f"Test {name}",
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        type_signature=type_signature,
        inputs=[IOSpec(name="x", type_desc="nat")],
        outputs=[IOSpec(name="y", type_desc="nat")],
    )


def _make_cdg(*node_ids: str, nodes: list[AlgorithmicNode] | None = None) -> CDGExport:
    if nodes is not None:
        return CDGExport(nodes=nodes, edges=[])
    built = [_make_atomic_node(nid, f"node_{nid}") for nid in node_ids]
    return CDGExport(nodes=built, edges=[])


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


def _make_template_match(children: list[dict], confidence: float = 0.85):
    """Build a fake TemplateMatch-like object for mocking."""

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
        _FakeChild(
            name=c["name"],
            description=c["description"],
            type_signature=c.get("type_signature", ""),
        )
        for c in children
    ]
    return _FakeMatch(
        example=_FakeExample(children=fake_children),
        confidence=confidence,
        source="verified_exemplar",
    )


def _events_of_type(event_type: str):
    """Return all telemetry events matching the given event_type."""
    return [ev for ev in get_event_log().events if ev.event_type == event_type]


def _make_llm_mock() -> AsyncMock:
    """Create an LLM mock that works with select_llm().complete()."""
    llm = AsyncMock()
    # select_llm(llm, key) returns llm itself when llm is not an LLMRouter,
    # so llm.complete(...) is what gets called.
    llm.complete = AsyncMock(return_value="[]")
    return llm


# ---------------------------------------------------------------------------
# Test 1: Retrieval hit skips deterministic and LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_retrieval_hit_skips_deterministic_and_llm():
    """When template_retriever returns confidence >= 0.7, retrieval split is used
    and neither deterministic nor LLM layers fire."""
    node = _make_atomic_node(
        "ret_node",
        "Decompose Matrix",
        description="Decompose matrix into factors",
        concept_type=ConceptType.ALGEBRA,
    )
    cdg = _make_cdg(nodes=[node])

    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="ret_node",
            statement="matrix decomposition",
            informal_desc="decompose matrix into factors",
        ),
        error_summaries=["no match found"],
        suggested_action=FailureAction.SPLIT,
    )

    retrieved_children = [
        {"name": "Factor A", "description": "compute first factor"},
        {"name": "Factor B", "description": "compute second factor"},
    ]
    template_retriever = AsyncMock()
    template_retriever.find_refinement_templates = AsyncMock(
        return_value=[_make_template_match(retrieved_children, confidence=0.85)]
    )

    llm = _make_llm_mock()

    updated = await refine_on_failure(
        failure, cdg, llm, template_retriever=template_retriever
    )

    # Should have 2 new sub-nodes
    children = [n for n in updated.nodes if n.parent_id == "ret_node"]
    assert len(children) == 2
    assert children[0].name == "Factor A"
    assert children[1].name == "Factor B"

    # LLM should NOT have been called
    llm.complete.assert_not_called()

    # Check telemetry for SPLIT_RETRIEVAL
    retrieval_events = _events_of_type("SPLIT_RETRIEVAL")
    assert len(retrieval_events) == 1
    assert retrieval_events[0].payload["node_id"] == "ret_node"
    assert retrieval_events[0].payload["confidence"] == 0.85

    # No deterministic or LLM events
    assert len(_events_of_type("SPLIT_DETERMINISTIC")) == 0
    assert len(_events_of_type("SPLIT_LLM")) == 0


# ---------------------------------------------------------------------------
# Test 2: Deterministic fallback when retrieval below threshold
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_deterministic_fallback_when_retrieval_below_threshold():
    """When retriever returns confidence < 0.7, deterministic pattern fires
    for an 'ecg bandpass filter' description."""
    node = AlgorithmicNode(
        node_id="ecg_node",
        name="Bandpass ECG Filter",
        description="Design and apply a stable bandpass filter to ECG samples.",
        concept_type=ConceptType.SIGNAL_FILTER,
        status=NodeStatus.ATOMIC,
        depth=0,
        inputs=[IOSpec(name="signal", type_desc="ndarray")],
        outputs=[IOSpec(name="filtered", type_desc="ndarray")],
    )
    cdg = _make_cdg(nodes=[node])

    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="ecg_node",
            statement="Bandpass raw ECG into cardiac frequency region",
            informal_desc="stable digital filter design and application",
        ),
        error_summaries=["Expected filtered_signal but got response tuple"],
        suggested_action=FailureAction.SPLIT,
    )

    # Retriever returns low-confidence match (below 0.7 threshold)
    low_conf_match = _make_template_match(
        [{"name": "X", "description": "x"}], confidence=0.5
    )
    template_retriever = AsyncMock()
    template_retriever.find_refinement_templates = AsyncMock(
        return_value=[low_conf_match]
    )

    llm = _make_llm_mock()

    updated = await refine_on_failure(
        failure, cdg, llm, template_retriever=template_retriever
    )

    parent = next(n for n in updated.nodes if n.node_id == "ecg_node")
    children = [n for n in updated.nodes if n.parent_id == "ecg_node"]
    assert parent.status == NodeStatus.DECOMPOSED
    assert [c.name for c in children] == ["Design Filter", "Apply Filter"]

    # LLM should NOT have been called
    llm.complete.assert_not_called()

    # Check telemetry for SPLIT_DETERMINISTIC (no SPLIT_RETRIEVAL)
    assert len(_events_of_type("SPLIT_RETRIEVAL")) == 0
    det_events = _events_of_type("SPLIT_DETERMINISTIC")
    assert len(det_events) == 1
    assert det_events[0].payload["node_id"] == "ecg_node"


# ---------------------------------------------------------------------------
# Test 3: LLM fallback when no deterministic match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_llm_fallback_when_no_deterministic_match():
    """When retriever returns nothing and description has no deterministic
    pattern, the LLM layer fires."""
    node = _make_atomic_node(
        "gen_node",
        "Generic Helper",
        description="generic helper that does something unique",
        concept_type=ConceptType.CUSTOM,
    )
    cdg = _make_cdg(nodes=[node])

    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="gen_node",
            statement="nat -> nat",
            informal_desc="generic helper that does something unique",
        ),
        error_summaries=["type mismatch"],
        suggested_action=FailureAction.SPLIT,
    )

    template_retriever = AsyncMock()
    template_retriever.find_refinement_templates = AsyncMock(return_value=[])

    llm = _make_llm_mock()
    llm.complete = AsyncMock(
        return_value='[{"name": "Sub Step 1", "description": "first part", "type_signature": "nat -> nat"}, '
        '{"name": "Sub Step 2", "description": "second part", "type_signature": "nat -> nat"}]'
    )

    updated = await refine_on_failure(
        failure, cdg, llm, template_retriever=template_retriever
    )

    parent = next(n for n in updated.nodes if n.node_id == "gen_node")
    children = [n for n in updated.nodes if n.parent_id == "gen_node"]
    assert parent.status == NodeStatus.DECOMPOSED
    assert len(children) == 2

    # Check telemetry for SPLIT_LLM (no retrieval or deterministic)
    assert len(_events_of_type("SPLIT_RETRIEVAL")) == 0
    assert len(_events_of_type("SPLIT_DETERMINISTIC")) == 0
    llm_events = _events_of_type("SPLIT_LLM")
    assert len(llm_events) == 1
    assert llm_events[0].payload["node_id"] == "gen_node"
    assert llm_events[0].payload["sub_node_count"] == 2


# ---------------------------------------------------------------------------
# Test 4: UNGROUNDABLE marks node REJECTED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ungroundable_action_marks_rejected():
    """FailureAction.UNGROUNDABLE sets the node status to REJECTED."""
    cdg = _make_cdg("ung_node")

    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="ung_node", statement="nat -> nat"),
        suggested_action=FailureAction.UNGROUNDABLE,
    )

    llm = _make_llm_mock()

    updated = await refine_on_failure(failure, cdg, llm)

    node = next(n for n in updated.nodes if n.node_id == "ung_node")
    assert node.status == NodeStatus.REJECTED
    assert "UNGROUNDABLE" in node.critic_notes


# ---------------------------------------------------------------------------
# Test 5: GENERALIZE strips algorithm names
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generalize_action_strips_algorithm_names():
    """FailureAction.GENERALIZE on 'Apply Dijkstra algorithm' strips
    'dijkstra', clears type_signature."""
    node = AlgorithmicNode(
        node_id="gen_step",
        name="Dijkstra Solver",
        description="Apply Dijkstra algorithm to find shortest paths in weighted graph",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        type_signature="Graph -> Distances",
        matched_primitive="dijkstra_shortest_path",
        primitive_binding_confidence=0.6,
        inputs=[IOSpec(name="g", type_desc="Graph")],
        outputs=[IOSpec(name="d", type_desc="Distances")],
    )
    cdg = _make_cdg(nodes=[node])

    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="gen_step", statement="shortest path"),
        error_summaries=["type mismatch"],
        suggested_action=FailureAction.GENERALIZE,
    )

    llm = _make_llm_mock()

    updated = await refine_on_failure(failure, cdg, llm)

    n = next(n for n in updated.nodes if n.node_id == "gen_step")
    # "dijkstra" should be stripped from description
    assert "dijkstra" not in n.description.lower()
    # type_signature should be cleared
    assert n.type_signature == ""
    # matched_primitive should be cleared
    assert n.matched_primitive is None
    assert n.primitive_binding_confidence == 0.0

    # Check telemetry for GENERALIZE_APPLIED
    gen_events = _events_of_type("GENERALIZE_APPLIED")
    assert len(gen_events) == 1
    assert gen_events[0].payload["node_id"] == "gen_step"


# ---------------------------------------------------------------------------
# Test 6: Multi-round orchestration with split + retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_orchestration_multi_round():
    """CDG with 3 leaves: Hunter fails leaf 3 in round 1, split produces
    2 sub-nodes, Hunter succeeds all in round 2. Assert rounds_used=2
    and all_matched=True."""
    nodes = [
        _make_atomic_node("leaf1", "Leaf One"),
        _make_atomic_node("leaf2", "Leaf Two"),
        _make_atomic_node("leaf3", "Leaf Three"),
    ]
    cdg = _make_cdg(nodes=nodes)

    call_round: dict[str, int] = {}

    async def mock_find_match(pdg_node):
        pid = pdg_node.predicate_id
        call_round.setdefault(pid, 0)
        call_round[pid] += 1

        # leaf1 and leaf2 always succeed
        if pid in ("leaf1", "leaf2"):
            return _make_match_result(pid, True)

        # leaf3 fails the first time
        if pid == "leaf3" and call_round[pid] == 1:
            return _make_match_result(pid, False)

        # Sub-nodes of leaf3 and leaf3 itself succeed on retry
        return _make_match_result(pid, True)

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(side_effect=mock_find_match)

    llm = _make_llm_mock()
    llm.complete = AsyncMock(
        return_value='[{"name": "Sub A", "description": "sub a part", "type_signature": "nat -> nat"}, '
        '{"name": "Sub B", "description": "sub b part", "type_signature": "nat -> nat"}]'
    )

    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=llm,
        max_rounds=3,
    )

    assert result.rounds_used == 2
    # The original leaf3 failure remains in match_results (its predicate_id
    # is never re-submitted since the node became DECOMPOSED). The sub-nodes
    # are the ones that get matched in round 2.
    sub_results = [
        mr for mr in result.match_results if mr.pdg_node.predicate_id.startswith("leaf3_sub")
    ]
    assert len(sub_results) == 2
    assert all(mr.success for mr in sub_results)
    # leaf1 and leaf2 matched in round 1
    for pid in ("leaf1", "leaf2"):
        mr = next(m for m in result.match_results if m.pdg_node.predicate_id == pid)
        assert mr.success
    assert len(result.ungroundable) == 0


# ---------------------------------------------------------------------------
# Test 7: Max rounds reached populates ungroundable
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_orchestration_max_rounds_reached():
    """When Hunter always fails and max_rounds=2, ungroundable should
    contain the failing node IDs."""
    node = _make_atomic_node("fail_node", "Always Fails")
    cdg = _make_cdg(nodes=[node])

    async def mock_find_match(pdg_node):
        return _make_match_result(pdg_node.predicate_id, False)

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(side_effect=mock_find_match)

    llm = _make_llm_mock()
    llm.complete = AsyncMock(
        return_value='[{"name": "StillFails1", "description": "part one", "type_signature": "nat -> nat"}, '
        '{"name": "StillFails2", "description": "part two", "type_signature": "nat -> nat"}]'
    )

    result = await run_orchestration(
        cdg,
        hunter_agent=hunter,
        llm=llm,
        max_rounds=2,
    )

    assert result.rounds_used == 2
    assert result.all_matched is False
    assert len(result.ungroundable) > 0

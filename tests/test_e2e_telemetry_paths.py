"""End-to-end tests verifying each execution path emits expected telemetry events."""

import asyncio
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
from ageom.telemetry import (
    PipelineEvent,
    get_event_log,
    log_event,
    reset_telemetry_runtime,
    telemetry_scope,
)
from ageom.types import (
    CandidateMatch,
    Declaration,
    FailureAction,
    MatchFailureReport,
    MatchResult,
    PDGNode,
    Prover,
)


@pytest.fixture(autouse=True)
def clean_telemetry():
    """Reset telemetry state before each test."""
    reset_telemetry_runtime()
    yield
    reset_telemetry_runtime()


# ---------------------------------------------------------------------------
# Helpers (mirrors test_orchestrator.py patterns)
# ---------------------------------------------------------------------------


def _make_atomic_node(node_id: str, name: str, **kwargs) -> AlgorithmicNode:
    defaults = dict(
        node_id=node_id,
        name=name,
        description=f"Test {name}",
        concept_type=ConceptType.SORTING,
        status=NodeStatus.ATOMIC,
        type_signature="nat -> nat",
        inputs=[IOSpec(name="x", type_desc="nat")],
        outputs=[IOSpec(name="y", type_desc="nat")],
    )
    defaults.update(kwargs)
    return AlgorithmicNode(**defaults)


def _make_cdg(*node_ids: str) -> CDGExport:
    nodes = [_make_atomic_node(nid, f"node_{nid}") for nid in node_ids]
    return CDGExport(nodes=nodes, edges=[])


def _make_match_result(node_id: str, success: bool) -> MatchResult:
    decl = Declaration(name=f"decl_{node_id}", type_signature="nat -> nat")
    candidate = CandidateMatch(declaration=decl, score=0.9, retrieval_method="embedding")
    vr = __import__("ageom.types", fromlist=["VerificationResult"]).VerificationResult(
        candidate=candidate, verified=success
    )
    return MatchResult(
        pdg_node=PDGNode(predicate_id=node_id, statement="nat -> nat"),
        verified_match=vr if success else None,
        all_candidates=[candidate],
        all_verifications=[vr],
    )


def _make_template_match(children, confidence=0.85):
    """Create a mock TemplateMatch with the given children and confidence."""
    from dataclasses import dataclass as _dc

    @_dc
    class _Child:
        name: str
        description: str
        type_signature: str

    @_dc
    class _Example:
        children: list

    @_dc
    class _Match:
        example: _Example
        confidence: float
        source: str

    fake_children = [
        _Child(name=c["name"], description=c["description"], type_signature=c.get("type_signature", ""))
        for c in children
    ]
    return _Match(
        example=_Example(children=fake_children),
        confidence=confidence,
        source="verified_exemplar",
    )


def _events_of_type(event_type: str) -> list[PipelineEvent]:
    return [e for e in get_event_log().events if e.event_type == event_type]


# ---------------------------------------------------------------------------
# 1. SPLIT_RETRIEVAL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refinement_split_retrieval_emits_event():
    """refine_on_failure with template_retriever returning confidence=0.85 emits SPLIT_RETRIEVAL."""
    node = AlgorithmicNode(
        node_id="ret_step",
        name="Matrix Decompose",
        description="Decompose matrix into lower-upper factors",
        concept_type=ConceptType.ALGEBRA,
        status=NodeStatus.ATOMIC,
        depth=0,
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(
            predicate_id="ret_step",
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

    await refine_on_failure(failure, cdg, llm, template_retriever=template_retriever)

    events = _events_of_type("SPLIT_RETRIEVAL")
    assert len(events) == 1
    assert events[0].payload["node_id"] == "ret_step"
    assert events[0].payload["confidence"] == 0.85
    assert events[0].payload["sub_node_count"] == 2


# ---------------------------------------------------------------------------
# 2. SPLIT_DETERMINISTIC
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refinement_split_deterministic_emits_event():
    """Node matching 'ecg bandpass filter' triggers deterministic split and emits SPLIT_DETERMINISTIC."""
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

    await refine_on_failure(failure, cdg, llm)

    events = _events_of_type("SPLIT_DETERMINISTIC")
    assert len(events) == 1
    assert events[0].payload["node_id"] == "filter_step"
    assert events[0].payload["sub_node_count"] >= 2


# ---------------------------------------------------------------------------
# 3. SPLIT_LLM
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_refinement_split_llm_emits_event():
    """No retriever, no pattern match, LLM returns JSON -> emits SPLIT_LLM."""
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

    await refine_on_failure(failure, cdg, llm)

    events = _events_of_type("SPLIT_LLM")
    assert len(events) == 1
    assert events[0].payload["node_id"] == "a"
    assert events[0].payload["sub_node_count"] >= 1


# ---------------------------------------------------------------------------
# 4. GENERALIZE_APPLIED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generalize_emits_event():
    """FailureAction.GENERALIZE on a node emits GENERALIZE_APPLIED."""
    node = AlgorithmicNode(
        node_id="solve_step",
        name="Dijkstra Solver",
        description="Use Dijkstra's algorithm to find shortest paths in weighted graph",
        concept_type=ConceptType.GRAPH_OPTIMIZATION,
        status=NodeStatus.ATOMIC,
        type_signature="Graph -> Distances",
    )
    cdg = CDGExport(nodes=[node], edges=[])
    failure = MatchFailureReport(
        pdg_node=PDGNode(predicate_id="solve_step", statement="shortest path"),
        error_summaries=["type mismatch"],
        suggested_action=FailureAction.GENERALIZE,
    )
    llm = AsyncMock()

    await refine_on_failure(failure, cdg, llm)

    events = _events_of_type("GENERALIZE_APPLIED")
    assert len(events) == 1
    assert events[0].payload["node_id"] == "solve_step"


# ---------------------------------------------------------------------------
# 5. log_event populates all fields
# ---------------------------------------------------------------------------


def test_log_event_populates_all_fields():
    """Direct log_event() call populates all PipelineEvent fields correctly."""
    ev = log_event(
        "test_round",
        "test_phase",
        "TEST_EVENT",
        node_id="n1",
        payload={"key": "value"},
        duration_ms=42.5,
        run_id="run_abc",
        stage="my_stage",
    )

    events = get_event_log().events
    assert len(events) == 1
    stored = events[0]

    assert stored.round == "test_round"
    assert stored.phase == "test_phase"
    assert stored.event_type == "TEST_EVENT"
    assert stored.node_id == "n1"
    assert stored.payload == {"key": "value"}
    assert stored.duration_ms == 42.5
    assert stored.run_id == "run_abc"
    assert stored.stage == "my_stage"
    assert stored.timestamp > 0

    # Returned event is the same object
    assert ev is stored


# ---------------------------------------------------------------------------
# 6. telemetry_scope sets context
# ---------------------------------------------------------------------------


def test_telemetry_scope_sets_context():
    """Inside telemetry_scope(), log_event inherits run_id and stage."""
    with telemetry_scope(run_id="scope_run", stage="scope_stage"):
        log_event("r", "p", "SCOPED_EVENT")

    events = _events_of_type("SCOPED_EVENT")
    assert len(events) == 1
    assert events[0].run_id == "scope_run"
    assert events[0].stage == "scope_stage"


# ---------------------------------------------------------------------------
# 7. run_orchestration emits round events
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestration_emits_round_events():
    """run_orchestration with a successful hunter emits ROUND_START and HUNTER_ROUND_DONE."""
    cdg = _make_cdg("a", "b")

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(
        side_effect=[
            _make_match_result("a", True),
            _make_match_result("b", True),
        ]
    )
    llm = AsyncMock()

    await run_orchestration(cdg, hunter_agent=hunter, llm=llm, max_rounds=3)

    round_starts = _events_of_type("ROUND_START")
    assert len(round_starts) >= 1
    assert round_starts[0].payload["round_num"] == 1

    hunter_done = _events_of_type("HUNTER_ROUND_DONE")
    assert len(hunter_done) >= 1
    assert hunter_done[0].payload["matches_succeeded"] == 2
    assert hunter_done[0].payload["matches_failed"] == 0

    # Should also emit ORCHESTRATION_DONE
    done = _events_of_type("ORCHESTRATION_DONE")
    assert len(done) == 1
    assert done[0].payload["rounds_used"] == 1


# ---------------------------------------------------------------------------
# 8. run_orchestration max_rounds emits MAX_ROUNDS_REACHED
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_orchestration_max_rounds_emits_event():
    """Hunter always fails, max_rounds=1 -> emits MAX_ROUNDS_REACHED."""
    cdg = _make_cdg("a")

    hunter = AsyncMock()
    hunter.find_match = AsyncMock(
        side_effect=lambda pdg_node: _make_match_result(pdg_node.predicate_id, False)
    )
    llm = AsyncMock()

    result = await run_orchestration(cdg, hunter_agent=hunter, llm=llm, max_rounds=1)

    max_rounds_events = _events_of_type("MAX_ROUNDS_REACHED")
    assert len(max_rounds_events) == 1
    assert "a" in max_rounds_events[0].payload["ungroundable"]

    # Also check ORCHESTRATION_DONE is emitted
    done = _events_of_type("ORCHESTRATION_DONE")
    assert len(done) == 1
    assert len(done[0].payload["ungroundable"]) >= 1

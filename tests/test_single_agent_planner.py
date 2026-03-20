from __future__ import annotations

from types import SimpleNamespace

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.orchestrator import OrchestratorResult
from sciona.services import (
    ArchitectDecomposeRequest,
    HunterBatchMatchResult,
    OrchestrationRequest,
    SingleAgentPlanner,
)
from sciona.types import FailureAction, MatchFailureReport, PDGNode, Prover


class _FakeHunterService:
    def __init__(self, *, direct_match, batch_result, direct_result) -> None:
        self._direct_match = direct_match
        self._batch_result = batch_result
        self._direct_result = direct_result
        self.goal_calls = 0
        self.batch_calls = 0

    async def match_goal(self, request):
        self.goal_calls += 1
        if isinstance(self._direct_match, list):
            index = min(self.goal_calls - 1, len(self._direct_match) - 1)
            return self._direct_match[index]
        return self._direct_match

    async def match_batch(self, request):
        self.batch_calls += 1
        if isinstance(self._batch_result, list):
            index = min(self.batch_calls - 1, len(self._batch_result) - 1)
            return self._batch_result[index]
        return self._batch_result

    def direct_match_result(self, *args, **kwargs):
        return self._direct_result


class _FakeArchitectService:
    def __init__(self, cdg) -> None:
        self._cdg = cdg
        self.calls = 0

    async def decompose(self, request: ArchitectDecomposeRequest):
        self.calls += 1
        return SimpleNamespace(goal=request.goal, cdg=self._cdg)


class _FakeOrchestratorService:
    def __init__(self, result) -> None:
        self._result = result
        self.calls = 0
        self.requests: list[OrchestrationRequest] = []

    async def run(self, request: OrchestrationRequest):
        self.calls += 1
        self.requests.append(request)
        return self._result


@pytest.mark.asyncio
async def test_single_agent_planner_returns_direct_result_without_decomposition():
    direct_result = OrchestratorResult(cdg=SimpleNamespace(), match_results=[], rounds_used=1)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=True),
        batch_result=HunterBatchMatchResult(match_results=[], failures=[], ungroundable=[]),
        direct_result=direct_result,
    )
    architect_called = False

    async def _architect_factory():
        nonlocal architect_called
        architect_called = True
        return _FakeArchitectService(SimpleNamespace())

    orchestrator = _FakeOrchestratorService(direct_result)

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Detect heart rate from ECG")

    assert result.execution_path == "single_agent_direct"
    assert result.result is direct_result
    assert [step.action for step in result.steps] == ["direct_match"]
    assert [step.action for step in result.state.tool_trace] == ["direct_match"]
    assert result.state.policy.direct_grounding_enabled is True
    assert result.state.policy.decomposition_mode == "single_pass"
    assert result.state.policy.retrieval_intensity == "light"
    assert result.state.policy.repair_policy == "bounded"
    assert result.state.budget.steps_used == 1
    assert result.state.budget.max_steps == 6
    assert result.state.verification_status == "verified"
    assert result.state.termination_reason == "direct_verified"
    assert result.state.open_failures == []
    assert result.state.artifacts == {
        "cdg": "direct_goal_cdg",
        "match_results": "direct_match_result",
    }
    assert result.state.artifact_mutations == {"cdg": 1, "match_results": 1}
    assert set(result.state.tool_metrics) == {"hunter.match_goal"}
    assert result.state.tool_metrics["hunter.match_goal"]["dispatches"] == 1
    assert result.state.tool_metrics["hunter.match_goal"]["latency_ms_total"] >= 0.0
    assert result.state.escalation_events == []
    assert result.state.attempt_history == ["direct_match"]
    assert hunter.goal_calls == 1
    assert hunter.batch_calls == 0
    assert architect_called is False
    assert orchestrator.calls == 0


@pytest.mark.asyncio
async def test_single_agent_planner_returns_structured_result_without_escalation():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Detect Heart Rate",
                description="Detect heart rate from ECG samples.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="np.ndarray -> float",
            )
        ],
        edges=[],
        metadata={},
    )
    structured_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=1)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=HunterBatchMatchResult(match_results=[], failures=[], ungroundable=[]),
        direct_result=structured_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(structured_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Detect heart rate from ECG")

    assert result.execution_path == "single_agent_structured"
    assert result.result.cdg is cdg
    assert [step.action for step in result.steps] == [
        "direct_match",
        "decompose",
        "match_decomposed",
    ]
    assert [step.status for step in result.state.tool_trace] == [
        "failed",
        "completed",
        "completed",
    ]
    assert result.state.budget.steps_used == 3
    assert result.state.budget.max_steps == 6
    assert result.state.verification_status == "verified"
    assert result.state.termination_reason == "structured_verified"
    assert result.state.open_failures == []
    assert result.state.artifacts == {
        "cdg": "architect_decompose",
        "match_results": "hunter_batch_match",
    }
    assert result.state.artifact_mutations == {"cdg": 1, "match_results": 1}
    assert result.state.attempt_history == [
        "direct_match",
        "decompose",
        "match_decomposed",
    ]
    assert hunter.goal_calls == 1
    assert hunter.batch_calls == 1
    assert architect.calls == 1
    assert orchestrator.calls == 0


@pytest.mark.asyncio
async def test_single_agent_planner_escalates_after_unresolved_single_pass():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Detect Heart Rate",
                description="Detect heart rate from ECG samples.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="np.ndarray -> float",
            )
        ],
        edges=[],
        metadata={},
    )
    orchestrated_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=2)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=HunterBatchMatchResult(
            match_results=[],
            failures=[],
            ungroundable=["n1"],
        ),
        direct_result=orchestrated_result,
    )
    architect = _FakeArchitectService(cdg)

    async def _architect_factory():
        return architect

    orchestrator = _FakeOrchestratorService(orchestrated_result)

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Detect heart rate from ECG")

    assert result.execution_path == "single_agent_escalated"
    assert result.result is orchestrated_result
    assert [step.action for step in result.steps] == [
        "direct_match",
        "decompose",
        "match_decomposed",
        "escalate_orchestration",
    ]
    assert [step.status for step in result.state.tool_trace] == [
        "failed",
        "completed",
        "partial",
        "completed",
    ]
    assert result.state.budget.steps_used == 4
    assert result.state.budget.max_steps == 6
    assert result.state.verification_status == "verified"
    assert result.state.termination_reason == "escalated_after_unresolved_leaves"
    assert result.state.open_failures == []
    assert result.state.artifacts == {
        "cdg": "architect_decompose",
        "match_results": "orchestrated_match_results",
        "orchestration": "run_orchestration",
    }
    assert result.state.artifact_mutations == {
        "cdg": 1,
        "match_results": 2,
        "orchestration": 1,
    }
    assert result.state.attempt_history == [
        "direct_match",
        "decompose",
        "match_decomposed",
        "escalate_orchestration",
    ]
    assert hunter.goal_calls == 1
    assert hunter.batch_calls == 1
    assert architect.calls == 1
    assert orchestrator.calls == 1
    assert orchestrator.requests[0].cdg is cdg


@pytest.mark.asyncio
async def test_single_agent_planner_skips_direct_match_for_compound_goals():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Bandpass ECG",
                description="Bandpass the ECG waveform.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="np.ndarray -> np.ndarray",
            )
        ],
        edges=[],
        metadata={},
    )
    structured_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=1)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=True),
        batch_result=HunterBatchMatchResult(match_results=[], failures=[], ungroundable=[]),
        direct_result=structured_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(structured_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Bandpass ECG and then detect heart rate")

    assert result.execution_path == "single_agent_structured"
    assert [step.action for step in result.steps] == ["decompose", "match_decomposed"]
    assert result.state.policy.direct_grounding_enabled is False
    assert result.state.policy.decomposition_mode == "selective_redecompose"
    assert result.state.policy.retrieval_intensity == "standard"
    assert result.state.policy.partial_accept_enabled is True
    assert result.state.policy.repair_policy == "bounded"
    assert result.state.verification_status == "verified"
    assert result.state.termination_reason == "structured_verified"
    assert hunter.goal_calls == 0
    assert hunter.batch_calls == 1
    assert architect.calls == 1
    assert orchestrator.calls == 0


@pytest.mark.asyncio
async def test_single_agent_planner_accepts_high_coverage_partial_result():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Stage One",
                description="First stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="A -> B",
            ),
            AlgorithmicNode(
                node_id="n2",
                name="Stage Two",
                description="Second stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="B -> C",
            ),
            AlgorithmicNode(
                node_id="n3",
                name="Stage Three",
                description="Third stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="C -> D",
            ),
            AlgorithmicNode(
                node_id="n4",
                name="Stage Four",
                description="Fourth stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="D -> E",
            ),
        ],
        edges=[],
        metadata={},
    )
    partial_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=1)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=HunterBatchMatchResult(
            match_results=[
                SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n1")),
                SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n2")),
                SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n3")),
            ],
            failures=[],
            ungroundable=["n4"],
        ),
        direct_result=partial_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(partial_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Detect heart rate from ECG")

    assert result.execution_path == "single_agent_partial"
    assert [step.action for step in result.steps] == [
        "direct_match",
        "decompose",
        "match_decomposed",
        "partial_accept",
    ]
    assert result.state.verification_status == "partial_verified"
    assert result.state.termination_reason == "partial_accept"
    assert result.state.open_failures == ["n4"]
    assert orchestrator.calls == 0


@pytest.mark.asyncio
async def test_single_agent_planner_retries_retrieval_before_escalation():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Stage One",
                description="First stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="A -> B",
            ),
            AlgorithmicNode(
                node_id="n2",
                name="Stage Two",
                description="Second stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="B -> C",
            ),
        ],
        edges=[],
        metadata={},
    )
    structured_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=1)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=[
            HunterBatchMatchResult(
                match_results=[
                    SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n1"))
                ],
                failures=[],
                ungroundable=["n2"],
            ),
            HunterBatchMatchResult(
                match_results=[
                    SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n2"))
                ],
                failures=[],
                ungroundable=[],
            ),
        ],
        direct_result=structured_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(structured_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run(
        "Bandpass ECG signal and then detect heart rate from filtered waveform"
    )

    assert result.execution_path == "single_agent_retried"
    assert [step.action for step in result.steps] == [
        "decompose",
        "match_decomposed",
        "retry_retrieval",
    ]
    assert result.state.policy.retrieval_intensity == "standard"
    assert result.state.verification_status == "verified"
    assert result.state.termination_reason == "retrieval_retry_verified"
    assert hunter.goal_calls == 0
    assert hunter.batch_calls == 2
    assert orchestrator.calls == 0


@pytest.mark.asyncio
async def test_single_agent_planner_single_pass_policy_skips_selective_redecompose():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="filter_step",
                name="Bandpass ECG Filter",
                description="Design and apply a stable bandpass filter to ECG samples.",
                concept_type=ConceptType.SIGNAL_FILTER,
                status=NodeStatus.ATOMIC,
                type_signature="np.ndarray -> np.ndarray",
            )
        ],
        edges=[],
        metadata={},
    )
    orchestrated_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=2)
    unresolved = HunterBatchMatchResult(
        match_results=[],
        failures=[
            MatchFailureReport(
                pdg_node=PDGNode(
                    predicate_id="filter_step",
                    statement="Bandpass raw ECG into cardiac frequency region",
                    informal_desc="stable digital filter design and application",
                ),
                error_summaries=["Expected filtered_signal but got response tuple"],
                suggested_action=FailureAction.SPLIT,
            )
        ],
        ungroundable=["filter_step"],
    )
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=[unresolved, unresolved],
        direct_result=orchestrated_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(orchestrated_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Bandpass ECG filter design for stable waveform preprocessing")

    assert result.execution_path == "single_agent_escalated"
    assert result.state.policy.decomposition_mode == "single_pass"
    assert [step.action for step in result.steps] == [
        "decompose",
        "match_decomposed",
        "retry_retrieval",
        "escalate_orchestration",
    ]
    assert result.state.escalation_events == [
        {
            "from": "direct_grounding",
            "to": "decomposition",
            "reason": "compound_goal_markers",
        },
        {
            "from": "decomposed_matching",
            "to": "orchestration",
            "reason": "unresolved_leaves_after_single_agent_attempts",
        },
    ]
    assert "selective_redecompose" not in result.state.attempt_history
    assert hunter.goal_calls == 0
    assert hunter.batch_calls == 2
    assert orchestrator.calls == 1


@pytest.mark.asyncio
async def test_single_agent_planner_uses_aggressive_retrieval_for_long_compound_goal():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Stage One",
                description="First stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="A -> B",
            )
        ],
        edges=[],
        metadata={},
    )
    structured_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=1)
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=HunterBatchMatchResult(match_results=[], failures=[], ungroundable=[]),
        direct_result=structured_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(structured_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run(
        "Bandpass ECG and then detect heart rate while comparing baseline drift before final reporting"
    )

    assert result.execution_path == "single_agent_structured"
    assert result.state.policy.direct_grounding_enabled is False
    assert result.state.policy.retrieval_intensity == "aggressive"
    assert result.state.policy.repair_policy == "until_verified"


@pytest.mark.asyncio
async def test_single_agent_planner_until_verified_policy_skips_partial_accept():
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="n1",
                name="Stage One",
                description="First stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="A -> B",
            ),
            AlgorithmicNode(
                node_id="n2",
                name="Stage Two",
                description="Second stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="B -> C",
            ),
            AlgorithmicNode(
                node_id="n3",
                name="Stage Three",
                description="Third stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="C -> D",
            ),
            AlgorithmicNode(
                node_id="n4",
                name="Stage Four",
                description="Fourth stage.",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
                type_signature="D -> E",
            ),
        ],
        edges=[],
        metadata={},
    )
    orchestrated_result = OrchestratorResult(cdg=cdg, match_results=[], rounds_used=2)
    unresolved = HunterBatchMatchResult(
        match_results=[
            SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n1")),
            SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n2")),
            SimpleNamespace(success=True, pdg_node=SimpleNamespace(predicate_id="n3")),
        ],
        failures=[],
        ungroundable=["n4"],
    )
    hunter = _FakeHunterService(
        direct_match=SimpleNamespace(success=False),
        batch_result=[unresolved, unresolved, unresolved],
        direct_result=orchestrated_result,
    )
    architect = _FakeArchitectService(cdg)
    orchestrator = _FakeOrchestratorService(orchestrated_result)

    async def _architect_factory():
        return architect

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrator=orchestrator,
        llm=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run(
        "Bandpass ECG and then detect heart rate while comparing baseline drift before final reporting"
    )

    assert result.execution_path == "single_agent_escalated"
    assert "partial_accept" not in [step.action for step in result.steps]
    assert [step.action for step in result.steps] == [
        "decompose",
        "match_decomposed",
        "retry_retrieval",
        "retry_retrieval",
        "escalate_orchestration",
    ]
    assert result.state.policy.repair_policy == "until_verified"
    assert orchestrator.calls == 1

from __future__ import annotations

from types import SimpleNamespace

import pytest

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from ageom.orchestrator import OrchestratorResult
from ageom.services import (
    ArchitectDecomposeRequest,
    HunterBatchMatchResult,
    SingleAgentPlanner,
)
from ageom.types import Prover


class _FakeHunterService:
    def __init__(self, *, direct_match, batch_result, direct_result) -> None:
        self._direct_match = direct_match
        self._batch_result = batch_result
        self._direct_result = direct_result
        self.goal_calls = 0
        self.batch_calls = 0

    async def match_goal(self, request):
        self.goal_calls += 1
        return self._direct_match

    async def match_batch(self, request):
        self.batch_calls += 1
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

    async def _orchestrate(*args, **kwargs):
        raise AssertionError("orchestration should not run on direct success")

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrate=_orchestrate,
        llm=object(),
        hunter_agent=object(),
        prover=Prover.PYTHON,
        max_rounds=2,
        hunter_concurrency=1,
    )

    result = await planner.run("Detect heart rate from ECG")

    assert result.execution_path == "single_agent_direct"
    assert result.result is direct_result
    assert [step.action for step in result.steps] == ["direct_match"]
    assert hunter.goal_calls == 1
    assert hunter.batch_calls == 0
    assert architect_called is False


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

    orchestrate_calls = 0

    async def _orchestrate(*args, **kwargs):
        nonlocal orchestrate_calls
        orchestrate_calls += 1
        return orchestrated_result

    planner = SingleAgentPlanner(
        hunter=hunter,
        architect_factory=_architect_factory,
        orchestrate=_orchestrate,
        llm=object(),
        hunter_agent=object(),
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
    assert hunter.goal_calls == 1
    assert hunter.batch_calls == 1
    assert architect.calls == 1
    assert orchestrate_calls == 1

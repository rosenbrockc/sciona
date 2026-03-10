"""First-cut single-agent planner runtime built on explicit tool services."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any

from ageom.architect.handoff import to_pdg_nodes
from ageom.orchestrator import OrchestratorResult
from ageom.services.models import (
    ArchitectDecomposeRequest,
    HunterBatchMatchRequest,
    HunterDirectMatchRequest,
    OrchestrationRequest,
    PlannerRunResult,
    PlannerStep,
)
from ageom.types import Prover


class SingleAgentPlanner:
    """Deterministic planner scaffold over Architect and Hunter tools.

    The first implementation keeps planner policy explicit:
    direct goal grounding -> shallow decomposition -> orchestration escalation.
    This extracts tool boundaries now without hiding control flow in an LLM.
    """

    def __init__(
        self,
        *,
        hunter: Any,
        architect_factory: Callable[[], Awaitable[Any]],
        orchestrator: Any,
        llm: Any,
        prover: Prover,
        max_rounds: int,
        hunter_concurrency: int,
    ) -> None:
        self._hunter = hunter
        self._architect_factory = architect_factory
        self._orchestrator = orchestrator
        self._llm = llm
        self._prover = prover
        self._max_rounds = max_rounds
        self._hunter_concurrency = hunter_concurrency

    async def run(self, goal: str) -> PlannerRunResult:
        steps: list[PlannerStep] = []

        direct_match = await self._hunter.match_goal(
            HunterDirectMatchRequest(
                goal=goal,
                prover=self._prover,
                informal_desc="single-agent planner direct grounding attempt",
                context={
                    "execution_mode": "single_agent",
                    "single_agent_direct_path": "true",
                },
            )
        )
        steps.append(
            PlannerStep(
                action="direct_match",
                detail="Attempted direct Hunter grounding before decomposition.",
            )
        )
        if getattr(direct_match, "success", False):
            return PlannerRunResult(
                result=self._hunter.direct_match_result(
                    goal,
                    self._prover,
                    direct_match,
                    execution_mode="single_agent",
                    informal_desc="Single-agent planner direct retrieval without architect decomposition.",
                    context={
                        "execution_mode": "single_agent",
                        "single_agent_direct_path": "true",
                    },
                ),
                execution_path="single_agent_direct",
                steps=steps,
            )

        architect = await self._architect_factory()
        decompose_result = await architect.decompose(
            ArchitectDecomposeRequest(goal=goal)
        )
        steps.append(
            PlannerStep(
                action="decompose",
                detail="Fell back to Architect decomposition after direct match failure.",
            )
        )

        pdg_nodes = to_pdg_nodes(decompose_result.cdg, prover=self._prover, strict=False)
        batch_result = await self._hunter.match_batch(
            HunterBatchMatchRequest(pdg_nodes=pdg_nodes)
        )
        steps.append(
            PlannerStep(
                action="match_decomposed",
                detail=f"Matched {len(pdg_nodes)} decomposed leaves once.",
            )
        )
        if not batch_result.ungroundable:
            return PlannerRunResult(
                result=OrchestratorResult(
                    cdg=decompose_result.cdg,
                    match_results=batch_result.match_results,
                    rounds_used=1,
                    failures=batch_result.failures,
                    ungroundable=batch_result.ungroundable,
                ),
                execution_path="single_agent_structured",
                steps=steps,
            )

        orchestrated = await self._orchestrator.run(
            OrchestrationRequest(
                cdg=decompose_result.cdg,
                llm=self._llm,
                prover=self._prover,
                max_rounds=self._max_rounds,
                hunter_concurrency=self._hunter_concurrency,
            )
        )
        steps.append(
            PlannerStep(
                action="escalate_orchestration",
                detail="Escalated to full orchestration after single-pass matching left unresolved leaves.",
            )
        )
        return PlannerRunResult(
            result=orchestrated,
            execution_path="single_agent_escalated",
            steps=steps,
        )

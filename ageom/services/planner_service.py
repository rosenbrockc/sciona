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
    PlannerBudget,
    PlannerPolicy,
    PlannerRunResult,
    PlannerState,
    PlannerStep,
)
from ageom.telemetry import log_event
from ageom.types import Prover


class SingleAgentPlanner:
    """Deterministic planner scaffold over Architect and Hunter tools.

    The first implementation keeps planner policy explicit:
    direct goal grounding -> shallow decomposition -> orchestration escalation.
    This extracts tool boundaries now without hiding control flow in an LLM.
    """

    _COMPOSITE_GOAL_MARKERS = (
        " and ",
        " and then ",
        " before ",
        " after ",
        " while ",
        " both ",
        " compare ",
        " by ",
        " pipeline",
        " workflow",
    )
    _MAX_DIRECT_GOAL_TOKENS = 6

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
        state = PlannerState(
            goal=goal,
            policy=self._select_policy(goal),
            budget=PlannerBudget(max_steps=4),
        )
        log_event(
            "planner",
            "decision",
            "PLANNER_POLICY",
            payload={
                "goal": goal,
                "direct_grounding_enabled": state.policy.direct_grounding_enabled,
                "decomposition_mode": state.policy.decomposition_mode,
                "escalation_enabled": state.policy.escalation_enabled,
                "reason": self._policy_reason(goal, state.policy),
            },
        )

        if state.policy.direct_grounding_enabled:
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
            self._record_step(
                state,
                action="direct_match",
                detail="Attempted direct Hunter grounding before decomposition.",
                status="completed" if getattr(direct_match, "success", False) else "failed",
            )
            if getattr(direct_match, "success", False):
                state.current_focus = "goal_grounded"
                state.open_failures = []
                state.verification_status = "verified"
                state.termination_reason = "direct_verified"
                return self._complete(
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
                    state=state,
                )

            state.current_focus = "decomposition"
            state.open_failures = ["goal_0"]
            state.verification_status = "needs_decomposition"
        else:
            state.current_focus = "decomposition"
            state.verification_status = "policy_decompose_first"
            log_event(
                "planner",
                "decision",
                "PLANNER_ESCALATION",
                payload={
                    "from": "direct_grounding",
                    "to": "decomposition",
                    "reason": "compound_goal_markers",
                },
            )
        architect = await self._architect_factory()
        decompose_result = await architect.decompose(
            ArchitectDecomposeRequest(goal=goal)
        )
        self._record_step(
            state,
            action="decompose",
            detail=(
                "Fell back to Architect decomposition after direct match failure."
                if state.policy.direct_grounding_enabled
                else "Selected Architect decomposition first based on planner policy."
            ),
        )

        pdg_nodes = to_pdg_nodes(decompose_result.cdg, prover=self._prover, strict=False)
        batch_result = await self._hunter.match_batch(
            HunterBatchMatchRequest(pdg_nodes=pdg_nodes)
        )
        state.current_focus = "decomposed_matching"
        state.open_failures = list(batch_result.ungroundable)
        state.verification_status = (
            "verified" if not batch_result.ungroundable else "needs_refinement"
        )
        self._record_step(
            state,
            action="match_decomposed",
            detail=f"Matched {len(pdg_nodes)} decomposed leaves once.",
            status="completed" if not batch_result.ungroundable else "partial",
        )
        if not batch_result.ungroundable:
            state.current_focus = "goal_grounded"
            state.termination_reason = "structured_verified"
            return self._complete(
                result=OrchestratorResult(
                    cdg=decompose_result.cdg,
                    match_results=batch_result.match_results,
                    rounds_used=1,
                    failures=batch_result.failures,
                    ungroundable=batch_result.ungroundable,
                ),
                execution_path="single_agent_structured",
                state=state,
            )

        state.current_focus = "orchestration"
        orchestrated = await self._orchestrator.run(
            OrchestrationRequest(
                cdg=decompose_result.cdg,
                llm=self._llm,
                prover=self._prover,
                max_rounds=self._max_rounds,
                hunter_concurrency=self._hunter_concurrency,
            )
        )
        state.open_failures = list(orchestrated.ungroundable)
        state.verification_status = (
            "verified" if not orchestrated.ungroundable else "partial"
        )
        state.termination_reason = "escalated_after_unresolved_leaves"
        self._record_step(
            state,
            action="escalate_orchestration",
            detail="Escalated to full orchestration after single-pass matching left unresolved leaves.",
            status="completed" if not orchestrated.ungroundable else "partial",
        )
        state.current_focus = "goal_grounded" if not state.open_failures else "residual_failures"
        return self._complete(
            result=orchestrated,
            execution_path="single_agent_escalated",
            state=state,
        )

    def _record_step(
        self,
        state: PlannerState,
        *,
        action: str,
        detail: str,
        status: str = "completed",
    ) -> None:
        step = PlannerStep(action=action, detail=detail, status=status)
        state.tool_trace.append(step)
        state.budget.steps_used += 1
        log_event(
            "planner",
            "decision",
            "PLANNER_STEP",
            payload={
                "action": action,
                "detail": detail,
                "status": status,
                "goal": state.goal,
                "current_focus": state.current_focus,
                "open_failures": list(state.open_failures),
                "steps_used": state.budget.steps_used,
                "step_budget": state.budget.max_steps,
            },
        )

    def _complete(
        self,
        *,
        result: OrchestratorResult,
        execution_path: str,
        state: PlannerState,
    ) -> PlannerRunResult:
        log_event(
            "planner",
            "decision",
            "PLANNER_COMPLETED",
            payload={
                "execution_path": execution_path,
                "termination_reason": state.termination_reason,
                "verification_status": state.verification_status,
                "steps_used": state.budget.steps_used,
                "step_budget": state.budget.max_steps,
                "open_failures": list(state.open_failures),
                "policy": {
                    "direct_grounding_enabled": state.policy.direct_grounding_enabled,
                    "decomposition_mode": state.policy.decomposition_mode,
                    "escalation_enabled": state.policy.escalation_enabled,
                },
            },
        )
        return PlannerRunResult(
            result=result,
            execution_path=execution_path,
            steps=list(state.tool_trace),
            state=state,
        )

    def _select_policy(self, goal: str) -> PlannerPolicy:
        if self._is_compound_goal(goal) or self._goal_token_count(goal) > self._MAX_DIRECT_GOAL_TOKENS:
            return PlannerPolicy(direct_grounding_enabled=False)
        return PlannerPolicy()

    def _is_compound_goal(self, goal: str) -> bool:
        normalized = f" {goal.strip().lower()} "
        return any(marker in normalized for marker in self._COMPOSITE_GOAL_MARKERS)

    def _goal_token_count(self, goal: str) -> int:
        return len([token for token in goal.strip().split() if token])

    def _policy_reason(self, goal: str, policy: PlannerPolicy) -> str:
        if not policy.direct_grounding_enabled and self._is_compound_goal(goal):
            return "compound_goal_markers"
        if not policy.direct_grounding_enabled:
            return "goal_too_long_for_direct_grounding"
        return "direct_first_default"

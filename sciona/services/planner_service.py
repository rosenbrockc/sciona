"""First-cut single-agent planner runtime built on explicit tool services."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
import time
from typing import Any

from sciona.architect.handoff import to_pdg_nodes
from sciona.orchestrator import OrchestratorResult
from sciona.services.models import (
    ArchitectDecomposeRequest,
    HunterBatchMatchRequest,
    HunterBatchMatchResult,
    HunterDirectMatchRequest,
    MacroMatchRequest,
    OrchestrationRequest,
    PlannerBudget,
    PlannerPolicy,
    PlannerRunResult,
    PlannerState,
    PlannerStep,
)
from sciona.telemetry import log_event
from sciona.types import Prover


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
    _AGGRESSIVE_GOAL_TOKENS = 12

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
        artifact_retriever: Any | None = None,
    ) -> None:
        self._hunter = hunter
        self._architect_factory = architect_factory
        self._orchestrator = orchestrator
        self._llm = llm
        self._prover = prover
        self._max_rounds = max_rounds
        self._hunter_concurrency = hunter_concurrency
        self._artifact_retriever = artifact_retriever

    async def run(self, goal: str) -> PlannerRunResult:
        state = PlannerState(
            goal=goal,
            policy=self._select_policy(goal),
            budget=PlannerBudget(max_steps=6),
        )
        log_event(
            "planner",
            "decision",
            "PLANNER_POLICY",
            payload={
                "goal": goal,
                "direct_grounding_enabled": state.policy.direct_grounding_enabled,
                "decomposition_mode": state.policy.decomposition_mode,
                "retrieval_intensity": state.policy.retrieval_intensity,
                "escalation_enabled": state.policy.escalation_enabled,
                "repair_policy": state.policy.repair_policy,
                "reason": self._policy_reason(goal, state.policy),
            },
        )

        decompose_result = None
        if state.policy.direct_grounding_enabled:
            if self._artifact_retriever is not None:
                macro_match = await self._timed_tool_call(
                    state,
                    "artifact.match_goal",
                    self._artifact_retriever.match_goal,
                    MacroMatchRequest(goal=goal),
                )
                self._record_step(
                    state,
                    action="direct_macro_match",
                    detail="Attempted published macro-artifact retrieval before Hunter grounding.",
                    status="completed" if getattr(macro_match, "success", False) else "failed",
                )
                if getattr(macro_match, "success", False):
                    candidate = getattr(macro_match, "candidate", None)
                    candidate_cdg = getattr(candidate, "cdg", None)
                    if (
                        candidate is not None
                        and candidate_cdg is not None
                        and not bool(getattr(candidate, "terminal_on_match", True))
                    ):
                        materialized_cdg = (
                            candidate_cdg.model_copy(deep=True)
                            if hasattr(candidate_cdg, "model_copy")
                            else candidate_cdg
                        )
                        if hasattr(materialized_cdg, "metadata"):
                            materialized_cdg.metadata = dict(
                                getattr(materialized_cdg, "metadata", {}) or {}
                            )
                            materialized_cdg.metadata["goal"] = goal
                        state.current_focus = "macro_decomposition"
                        state.open_failures = []
                        state.artifacts["cdg"] = "macro_artifact_cdg"
                        self._bump_artifact(state, "cdg")
                        state.verification_status = "macro_selected"
                        decompose_result = SimpleNamespace(
                            goal=goal,
                            cdg=materialized_cdg,
                        )
                    else:
                        state.current_focus = "goal_grounded"
                        state.open_failures = []
                        state.artifacts["cdg"] = "macro_goal_cdg"
                        state.artifacts["macro_match"] = "artifact_direct_match"
                        self._bump_artifact(state, "cdg")
                        self._bump_artifact(state, "macro_match")
                        state.verification_status = "verified"
                        state.termination_reason = "macro_direct_verified"
                        return self._complete(
                            result=self._hunter.macro_match_result(
                                goal,
                                self._prover,
                                macro_match,
                                execution_mode="single_agent",
                                informal_desc="Single-agent planner direct macro-artifact retrieval without architect decomposition.",
                                context={
                                    "execution_mode": "single_agent",
                                    "single_agent_macro_direct_path": "true",
                                },
                            ),
                            execution_path="single_agent_macro_direct",
                            state=state,
                        )

            if decompose_result is None:
                direct_match = await self._timed_tool_call(
                    state,
                    "hunter.match_goal",
                    self._hunter.match_goal,
                    HunterDirectMatchRequest(
                        goal=goal,
                        prover=self._prover,
                        informal_desc="single-agent planner direct grounding attempt",
                        context={
                            "execution_mode": "single_agent",
                            "single_agent_direct_path": "true",
                        },
                    ),
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
                    state.artifacts["cdg"] = "direct_goal_cdg"
                    state.artifacts["match_results"] = "direct_match_result"
                    self._bump_artifact(state, "cdg")
                    self._bump_artifact(state, "match_results")
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
                self._record_escalation(
                    state,
                    from_focus="direct_grounding",
                    to_focus="decomposition",
                    reason="direct_match_failed",
                )
        else:
            state.current_focus = "decomposition"
            state.verification_status = "policy_decompose_first"
            self._record_escalation(
                state,
                from_focus="direct_grounding",
                to_focus="decomposition",
                reason="compound_goal_markers",
            )
        if decompose_result is None:
            architect = await self._architect_factory()
            decompose_result = await self._timed_tool_call(
                state,
                "architect.decompose",
                architect.decompose,
                ArchitectDecomposeRequest(goal=goal),
            )
            state.artifacts["cdg"] = "architect_decompose"
            self._bump_artifact(state, "cdg")
            self._record_step(
                state,
                action="decompose",
                detail=(
                    "Fell back to Architect decomposition after direct match failure."
                    if state.policy.direct_grounding_enabled
                    else "Selected Architect decomposition first based on planner policy."
                ),
            )
        else:
            self._record_step(
                state,
                action="macro_decompose",
                detail="Selected a macro CDG artifact and skipped Architect decomposition.",
            )

        pdg_nodes = to_pdg_nodes(decompose_result.cdg, prover=self._prover, strict=False)
        batch_result = await self._timed_tool_call(
            state,
            "hunter.match_batch",
            self._hunter.match_batch,
            HunterBatchMatchRequest(pdg_nodes=pdg_nodes),
        )
        state.current_focus = "decomposed_matching"
        state.open_failures = list(batch_result.ungroundable)
        state.artifacts["match_results"] = "hunter_batch_match"
        self._bump_artifact(state, "match_results")
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
            state.termination_reason = (
                "macro_structured_verified"
                if state.artifacts.get("cdg") == "macro_artifact_cdg"
                else "structured_verified"
            )
            return self._complete(
                result=OrchestratorResult(
                    cdg=decompose_result.cdg,
                    match_results=batch_result.match_results,
                    rounds_used=1,
                    failures=batch_result.failures,
                    ungroundable=batch_result.ungroundable,
                ),
                execution_path=(
                    "single_agent_macro_structured"
                    if state.artifacts.get("cdg") == "macro_artifact_cdg"
                    else "single_agent_structured"
                ),
                state=state,
            )

        retry_limit = self._retrieval_retry_limit(state.policy)
        if retry_limit > 0 and batch_result.ungroundable:
            retried = await self._retry_retrieval(
                state=state,
                pdg_nodes=pdg_nodes,
                batch_result=batch_result,
            )
            batch_result = retried
            if not batch_result.ungroundable:
                state.current_focus = "goal_grounded"
                state.open_failures = []
                state.termination_reason = "retrieval_retry_verified"
                state.verification_status = "verified"
                return self._complete(
                    result=OrchestratorResult(
                        cdg=decompose_result.cdg,
                        match_results=batch_result.match_results,
                        rounds_used=1,
                        failures=batch_result.failures,
                        ungroundable=batch_result.ungroundable,
                    ),
                    execution_path="single_agent_retried",
                    state=state,
                )

        # --- Partial result acceptance ---
        # If most leaves matched, accept the partial result without escalation
        if self._allows_partial_accept(state.policy):
            total_leaves = len(pdg_nodes)
            matched_leaves = total_leaves - len(batch_result.ungroundable)
            if total_leaves > 0 and matched_leaves / total_leaves >= 0.7:
                state.current_focus = "goal_grounded"
                state.termination_reason = "partial_accept"
                state.verification_status = "partial_verified"
                self._record_step(
                    state,
                    action="partial_accept",
                    detail=(
                        f"Accepted partial result: {matched_leaves}/{total_leaves} leaves matched "
                        f"({len(batch_result.ungroundable)} unresolved)."
                    ),
                )
                log_event(
                    "planner",
                    "decision",
                    "PLANNER_PARTIAL_ACCEPT",
                    payload={
                        "matched": matched_leaves,
                        "total": total_leaves,
                        "ungroundable": list(batch_result.ungroundable),
                    },
                )
                return self._complete(
                    result=OrchestratorResult(
                        cdg=decompose_result.cdg,
                        match_results=batch_result.match_results,
                        rounds_used=1,
                        failures=batch_result.failures,
                        ungroundable=batch_result.ungroundable,
                    ),
                    execution_path="single_agent_partial",
                    state=state,
                )

        # --- Selective re-decomposition ---
        # Instead of escalating the entire CDG, re-decompose only the failed leaves
        # and retry matching just those. This avoids the expense of full orchestration.
        if (
            self._allows_selective_redecompose(state.policy)
            and batch_result.failures
            and state.budget.steps_used < state.budget.max_steps - 1
        ):
            from sciona.orchestrator import (
                _deterministic_split_subnodes,
                _find_cdg_node,
                _apply_split_subnodes,
            )

            cdg = decompose_result.cdg
            split_count = 0
            for failure in batch_result.failures:
                original = _find_cdg_node(cdg, failure.pdg_node.predicate_id)
                sub_nodes = _deterministic_split_subnodes(failure, original)
                if sub_nodes and original is not None:
                    _apply_split_subnodes(cdg, original, sub_nodes)
                    split_count += 1

            if split_count > 0:
                state.artifacts["cdg"] = "selective_redecompose"
                self._bump_artifact(state, "cdg")
                self._record_step(
                    state,
                    action="selective_redecompose",
                    detail=f"Deterministically re-decomposed {split_count} failed leaves.",
                )
                # Re-match only the new sub-nodes
                new_pdg_nodes = to_pdg_nodes(cdg, prover=self._prover, strict=False)
                already_matched = {
                    mr.pdg_node.predicate_id
                    for mr in batch_result.match_results
                    if getattr(mr, "success", False)
                }
                retry_nodes = [n for n in new_pdg_nodes if n.predicate_id not in already_matched]

                if retry_nodes:
                    retry_result = await self._timed_tool_call(
                        state,
                        "hunter.match_batch",
                        self._hunter.match_batch,
                        HunterBatchMatchRequest(pdg_nodes=retry_nodes),
                    )
                    # Merge results
                    all_results = [
                        mr for mr in batch_result.match_results
                        if mr.pdg_node.predicate_id in already_matched
                    ] + retry_result.match_results
                    all_ungroundable = retry_result.ungroundable
                    all_failures = retry_result.failures
                    state.artifacts["match_results"] = "retry_match"
                    self._bump_artifact(state, "match_results")

                    self._record_step(
                        state,
                        action="retry_match",
                        detail=f"Retried {len(retry_nodes)} nodes after re-decomposition.",
                        status="completed" if not all_ungroundable else "partial",
                    )

                    if not all_ungroundable:
                        state.current_focus = "goal_grounded"
                        state.open_failures = []
                        state.termination_reason = "redecompose_verified"
                        state.verification_status = "verified"
                        return self._complete(
                            result=OrchestratorResult(
                                cdg=cdg,
                                match_results=all_results,
                                rounds_used=1,
                                failures=all_failures,
                                ungroundable=all_ungroundable,
                            ),
                            execution_path="single_agent_redecomposed",
                            state=state,
                        )

                    # Update state for potential escalation
                    state.open_failures = list(all_ungroundable)
                    batch_result = type(batch_result)(
                        match_results=all_results,
                        failures=all_failures,
                        ungroundable=all_ungroundable,
                    )

        state.current_focus = "orchestration"
        self._record_escalation(
            state,
            from_focus="decomposed_matching",
            to_focus="orchestration",
            reason="unresolved_leaves_after_single_agent_attempts",
        )
        orchestrated = await self._timed_tool_call(
            state,
            "orchestrator.run",
            self._orchestrator.run,
            OrchestrationRequest(
                cdg=decompose_result.cdg,
                llm=self._llm,
                prover=self._prover,
                max_rounds=self._max_rounds,
                hunter_concurrency=self._hunter_concurrency,
            ),
        )
        state.artifacts["orchestration"] = "run_orchestration"
        state.artifacts["match_results"] = "orchestrated_match_results"
        self._bump_artifact(state, "orchestration")
        self._bump_artifact(state, "match_results")
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
        state.attempt_history.append(action)
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
                "artifacts": dict(state.artifacts),
                "artifact_mutations": dict(state.artifact_mutations),
                "tool_metrics": {
                    name: dict(metrics) for name, metrics in state.tool_metrics.items()
                },
                "escalation_events": [dict(event) for event in state.escalation_events],
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
                "artifacts": dict(state.artifacts),
                "artifact_mutations": dict(state.artifact_mutations),
                "tool_metrics": {
                    name: dict(metrics) for name, metrics in state.tool_metrics.items()
                },
                "escalation_events": [dict(event) for event in state.escalation_events],
                "policy": {
                    "direct_grounding_enabled": state.policy.direct_grounding_enabled,
                    "decomposition_mode": state.policy.decomposition_mode,
                    "retrieval_intensity": state.policy.retrieval_intensity,
                    "escalation_enabled": state.policy.escalation_enabled,
                    "repair_policy": state.policy.repair_policy,
                    "partial_accept_enabled": state.policy.partial_accept_enabled,
                    "selective_redecompose_enabled": state.policy.selective_redecompose_enabled,
                },
            },
        )
        return PlannerRunResult(
            result=result,
            execution_path=execution_path,
            steps=list(state.tool_trace),
            state=state,
        )

    def _bump_artifact(self, state: PlannerState, artifact_name: str) -> None:
        state.artifact_mutations[artifact_name] = (
            int(state.artifact_mutations.get(artifact_name, 0) or 0) + 1
        )

    async def _timed_tool_call(
        self,
        state: PlannerState,
        tool_name: str,
        tool: Callable[..., Awaitable[Any]],
        *args: Any,
    ) -> Any:
        started = time.perf_counter()
        try:
            return await tool(*args)
        finally:
            self._record_tool_metric(
                state,
                tool_name,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )

    def _record_tool_metric(
        self,
        state: PlannerState,
        tool_name: str,
        *,
        latency_ms: float,
    ) -> None:
        current = state.tool_metrics.setdefault(
            tool_name,
            {
                "dispatches": 0,
                "latency_ms_total": 0.0,
                "avg_latency_ms": 0.0,
            },
        )
        current["dispatches"] = int(current.get("dispatches", 0) or 0) + 1
        current["latency_ms_total"] = float(
            current.get("latency_ms_total", 0.0) or 0.0
        ) + max(0.0, latency_ms)
        current["avg_latency_ms"] = (
            float(current["latency_ms_total"]) / max(1, int(current["dispatches"]))
        )

    def _record_escalation(
        self,
        state: PlannerState,
        *,
        from_focus: str,
        to_focus: str,
        reason: str,
    ) -> None:
        event = {
            "from": from_focus,
            "to": to_focus,
            "reason": reason,
        }
        state.escalation_events.append(event)
        log_event(
            "planner",
            "decision",
            "PLANNER_ESCALATION",
            payload={
                "goal": state.goal,
                "from": from_focus,
                "to": to_focus,
                "reason": reason,
                "steps_used": state.budget.steps_used,
                "step_budget": state.budget.max_steps,
            },
        )

    async def _retry_retrieval(
        self,
        *,
        state: PlannerState,
        pdg_nodes: list[Any],
        batch_result: HunterBatchMatchResult,
    ) -> HunterBatchMatchResult:
        pending_ids = set(batch_result.ungroundable)
        current = batch_result
        result_by_node = {
            str(mr.pdg_node.predicate_id): mr for mr in batch_result.match_results
        }
        failure_by_node = {
            str(failure.pdg_node.predicate_id): failure for failure in batch_result.failures
        }

        for attempt_idx in range(self._retrieval_retry_limit(state.policy)):
            if not pending_ids or state.budget.steps_used >= state.budget.max_steps - 1:
                break
            retry_nodes = [node for node in pdg_nodes if node.predicate_id in pending_ids]
            if not retry_nodes:
                break
            state.current_focus = "retrieval_retry"
            current = await self._timed_tool_call(
                state,
                "hunter.match_batch",
                self._hunter.match_batch,
                HunterBatchMatchRequest(pdg_nodes=retry_nodes),
            )
            for mr in current.match_results:
                result_by_node[str(mr.pdg_node.predicate_id)] = mr
            failure_by_node = {
                str(failure.pdg_node.predicate_id): failure for failure in current.failures
            }
            state.artifacts["match_results"] = "retrieval_retry"
            self._bump_artifact(state, "match_results")
            pending_ids = set(current.ungroundable)
            state.open_failures = sorted(pending_ids)
            state.verification_status = (
                "verified" if not pending_ids else "needs_refinement"
            )
            self._record_step(
                state,
                action="retry_retrieval",
                detail=(
                    f"Retried unresolved leaves ({attempt_idx + 1}/"
                    f"{self._retrieval_retry_limit(state.policy)})."
                ),
                status="completed" if not pending_ids else "partial",
            )

        ordered_results = [
            result_by_node[node.predicate_id]
            for node in pdg_nodes
            if node.predicate_id in result_by_node
        ]
        ordered_failures = [
            failure_by_node[node.predicate_id]
            for node in pdg_nodes
            if node.predicate_id in failure_by_node
        ]
        return HunterBatchMatchResult(
            match_results=ordered_results,
            failures=ordered_failures,
            ungroundable=sorted(pending_ids),
        )

    def _retrieval_retry_limit(self, policy: PlannerPolicy) -> int:
        if policy.retrieval_intensity == "aggressive":
            return 2
        if policy.retrieval_intensity == "standard":
            return 1
        return 0

    def _allows_partial_accept(self, policy: PlannerPolicy) -> bool:
        if not policy.partial_accept_enabled:
            return False
        return policy.repair_policy == "bounded"

    def _allows_selective_redecompose(self, policy: PlannerPolicy) -> bool:
        if not policy.selective_redecompose_enabled:
            return False
        if policy.decomposition_mode != "selective_redecompose":
            return False
        return policy.repair_policy in {"bounded", "until_verified"}

    def _select_policy(self, goal: str) -> PlannerPolicy:
        token_count = self._goal_token_count(goal)
        if self._is_compound_goal(goal):
            return PlannerPolicy(
                direct_grounding_enabled=False,
                decomposition_mode="selective_redecompose",
                retrieval_intensity=(
                    "aggressive" if token_count >= self._AGGRESSIVE_GOAL_TOKENS else "standard"
                ),
                repair_policy=(
                    "until_verified" if token_count >= self._AGGRESSIVE_GOAL_TOKENS else "bounded"
                ),
                partial_accept_enabled=True,
                selective_redecompose_enabled=True,
            )
        if token_count > self._MAX_DIRECT_GOAL_TOKENS:
            return PlannerPolicy(
                direct_grounding_enabled=False,
                decomposition_mode="single_pass",
                retrieval_intensity=(
                    "aggressive" if token_count >= self._AGGRESSIVE_GOAL_TOKENS else "standard"
                ),
                repair_policy="bounded",
            )
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

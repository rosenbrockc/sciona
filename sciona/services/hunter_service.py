"""Service wrapper around the Hunter retrieval agent."""

from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.orchestrator import OrchestratorResult
from sciona.services.models import (
    HunterBatchMatchRequest,
    HunterBatchMatchResult,
    HunterDirectMatchRequest,
)
from sciona.types import MatchFailureReport, PDGNode, Prover


def build_direct_goal_cdg(
    goal: str,
    prover: Prover,
    match_result: object,
    *,
    execution_mode: str,
    conceptual_summary: str,
) -> CDGExport:
    """Build a minimal one-node CDG for direct-match execution paths."""
    verified = getattr(match_result, "verified_match", None)
    verified_decl = getattr(getattr(verified, "candidate", None), "declaration", None)
    top_decl = None
    if getattr(match_result, "all_candidates", None):
        top_decl = getattr(match_result.all_candidates[0], "declaration", None)

    type_signature = ""
    matched_primitive = None
    if verified_decl is not None:
        type_signature = str(getattr(verified_decl, "type_signature", "") or "").strip()
        matched_primitive = str(getattr(verified_decl, "name", "") or "").strip() or None
    elif top_decl is not None:
        type_signature = str(getattr(top_decl, "type_signature", "") or "").strip()

    failure_notes = ""
    for verification in getattr(match_result, "all_verifications", []) or []:
        failure_notes = str(getattr(verification, "error_message", "") or "").strip()
        if failure_notes:
            break
    if not failure_notes:
        failure_notes = (
            "Direct match found no verified candidate."
            if getattr(match_result, "all_candidates", None)
            else "Direct match found no candidates."
        )

    success = bool(getattr(match_result, "success", False))
    node = AlgorithmicNode(
        node_id="goal_0",
        name="Direct Goal Match",
        description=goal,
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC if success else NodeStatus.BLOCKED,
        type_signature=type_signature,
        matched_primitive=matched_primitive,
        critic_notes=("Direct match succeeded." if success else failure_notes),
        conceptual_summary=conceptual_summary,
    )
    metadata = {
        "goal": goal,
        "prover": prover.value,
        "execution_mode": execution_mode,
        "rapid_direct_path": execution_mode == "rapid",
        "single_agent_direct_path": execution_mode == "single_agent",
        "num_nodes": 1,
        "num_edges": 0,
        "matched_directly": success,
    }
    if not success:
        metadata["architect_error"] = f"Direct match failed: {failure_notes}"
    return CDGExport(nodes=[node], edges=[], metadata=metadata)


class HunterService:
    """Stable service entrypoint for retrieval and verification operations."""

    def __init__(self, hunter: object) -> None:
        self._hunter = hunter

    async def match_goal(self, request: HunterDirectMatchRequest):
        node = PDGNode(
            predicate_id=request.predicate_id,
            statement=request.goal,
            informal_desc=request.informal_desc,
            prover=request.prover,
            context=request.context,
        )
        return await self._hunter.find_match(node)

    async def match_batch(
        self,
        request: HunterBatchMatchRequest,
    ) -> HunterBatchMatchResult:
        match_results = []
        failures = []
        ungroundable: list[str] = []
        for pdg_node in request.pdg_nodes:
            match_result = await self._hunter.find_match(pdg_node)
            match_results.append(match_result)
            if getattr(match_result, "success", False):
                continue
            failures.append(MatchFailureReport.from_match_result(match_result))
            ungroundable.append(pdg_node.predicate_id)
        return HunterBatchMatchResult(
            match_results=match_results,
            failures=failures,
            ungroundable=ungroundable,
        )

    def direct_match_result(
        self,
        goal: str,
        prover: Prover,
        match_result: object,
        *,
        execution_mode: str,
        informal_desc: str,
        context: dict[str, str] | None = None,
    ) -> OrchestratorResult:
        failures = []
        ungroundable: list[str] = []
        if not getattr(match_result, "success", False):
            failures.append(MatchFailureReport.from_match_result(match_result))
            ungroundable.append("goal_0")
        return OrchestratorResult(
            cdg=build_direct_goal_cdg(
                goal,
                prover,
                match_result,
                execution_mode=execution_mode,
                conceptual_summary=informal_desc,
            ),
            match_results=[match_result],
            rounds_used=1,
            failures=failures,
            ungroundable=ungroundable,
        )

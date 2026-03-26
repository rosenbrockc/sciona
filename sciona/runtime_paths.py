"""Shared runtime helpers for direct and structured execution paths."""

from __future__ import annotations

import uuid
from typing import Any

from sciona.architect.handoff import CDGExport, to_pdg_nodes
from sciona.architect.models import AlgorithmicNode, ConceptType, NodeStatus
from sciona.expansion_atoms.signal_event_rate_registry import (
    SIGNAL_EVENT_RATE_DECLARATIONS,
)
from sciona.orchestrator import OrchestratorResult
from sciona.services.hunter_service import (
    HunterBatchMatchRequest,
    HunterDirectMatchRequest,
    HunterService,
    build_direct_goal_cdg,
)
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)


def _matches_signal_event_rate_goal(goal: str) -> bool:
    lowered = goal.lower()
    signal_terms = ("signal", "waveform", "ecg", "ppg", "eeg", "sensor")
    detect_terms = ("detect", "peak", "event", "events")
    rate_terms = ("rate", "cadence", "rhythm")
    return (
        any(term in lowered for term in signal_terms)
        and any(term in lowered for term in detect_terms)
        and any(term in lowered for term in rate_terms)
    )


def _is_signal_event_rate_scaffold(cdg: Any) -> bool:
    atomic_nodes = [
        node
        for node in getattr(cdg, "nodes", [])
        if getattr(node, "status", None).value == "atomic"
    ]
    if not atomic_nodes:
        return False
    return all(
        node.matched_primitive in SIGNAL_EVENT_RATE_DECLARATIONS
        for node in atomic_nodes
    )


def _build_signal_event_rate_match_results(
    cdg: Any,
    prover: Prover,
) -> list[MatchResult]:
    match_results: list[MatchResult] = []
    for node in cdg.nodes:
        if getattr(node, "status", None).value != "atomic":
            continue
        primitive_name = str(node.matched_primitive or "").strip()
        decl_info = SIGNAL_EVENT_RATE_DECLARATIONS.get(primitive_name)
        if decl_info is None:
            continue
        declaration_name, type_signature, docstring = decl_info
        declaration = Declaration(
            name=declaration_name,
            type_signature=type_signature,
            docstring=docstring,
            conceptual_summary=node.description,
            source_lib="sciona.expansion_atoms.runtime_signal_event_rate",
            prover=prover,
        )
        candidate = CandidateMatch(
            declaration=declaration,
            score=1.0,
            retrieval_method="curated_signal_event_rate",
        )
        verification = VerificationResult(
            candidate=candidate,
            verified=True,
            verification_level=VerificationLevel.CONTRACT_CHECKED,
        )
        match_results.append(
            MatchResult(
                pdg_node=PDGNode(
                    predicate_id=node.node_id,
                    statement=node.name,
                    informal_desc=node.description,
                    prover=prover,
                    context={"curated_signal_event_rate": "true"},
                ),
                verified_match=verification,
                all_candidates=[candidate],
                all_verifications=[verification],
            )
        )
    return match_results


def _build_signal_event_rate_cdg(goal: str, prover: Prover) -> CDGExport:
    """Build a small deterministic CDG for signal event-rate estimation goals."""
    from sciona.architect.handoff import CDGExport
    from sciona.architect.skeletons import (
        get_skeleton,
        infer_boundary_ports,
        instantiate_skeleton,
    )

    skeleton = get_skeleton(ConceptType.SIGNAL_FILTER, variant="event_rate_estimation")
    if skeleton is None:
        raise RuntimeError("event_rate_estimation skeleton is not available")

    root_id = f"root_{uuid.uuid4().hex[:8]}"
    nodes, edges = instantiate_skeleton(skeleton, goal, parent_id=root_id, base_depth=0)
    nodes = [
        node.model_copy(
            update={
                "status": NodeStatus.ATOMIC if node.matched_primitive else node.status,
            }
        )
        for node in nodes
    ]
    root_inputs, root_outputs = infer_boundary_ports(nodes, edges)
    root = AlgorithmicNode(
        node_id=root_id,
        name=goal,
        description=goal,
        concept_type=ConceptType.SIGNAL_FILTER,
        inputs=root_inputs,
        outputs=root_outputs,
        status=NodeStatus.DECOMPOSED,
        children=[node.node_id for node in nodes],
        depth=0,
        conceptual_summary="Rapid-mode deterministic signal event-rate scaffold.",
    )

    return CDGExport(
        nodes=[root] + nodes,
        edges=edges,
        metadata={
            "goal": goal,
            "prover": prover.value,
            "execution_mode": "rapid",
            "rapid_direct_path": True,
            "single_agent_direct_path": False,
            "num_nodes": len(nodes) + 1,
            "num_edges": len(edges),
            "matched_directly": False,
            "rapid_signal_event_rate_path": True,
        },
    )


def _build_rapid_direct_cdg(
    goal: str,
    prover: Prover,
    match_result: object,
) -> CDGExport:
    """Build a minimal one-node CDG for rapid direct-match runs."""
    return build_direct_goal_cdg(
        goal,
        prover,
        match_result,
        execution_mode="rapid",
        conceptual_summary="Rapid-mode direct retrieval without architect decomposition.",
    )


async def _run_rapid_direct_match(
    goal: str,
    *,
    prover: Prover,
    hunter: Any,
) -> OrchestratorResult:
    """Run the rapid-mode direct Hunter path and wrap it in an orchestration result."""
    service = HunterService(hunter)
    if _matches_signal_event_rate_goal(goal):
        from sciona.telemetry import log_event as _log_event

        _log_event(
            "run_cmds",
            "fast_path",
            "DETERMINISTIC_FAST_PATH_FIRED",
            payload={"goal": goal[:100], "path": "signal_event_rate"},
        )
        _log_event(
            "run",
            "fast_path",
            "TEMPLATE_EQUIVALENT",
            payload={"exemplar": "signal_event_rate"},
        )
        cdg = _build_signal_event_rate_cdg(goal, prover)
        match_results = _build_signal_event_rate_match_results(cdg, prover)
        return OrchestratorResult(
            cdg=cdg,
            match_results=match_results,
            rounds_used=1,
            failures=[],
            ungroundable=[],
        )

    match_result = await service.match_goal(
        HunterDirectMatchRequest(
            goal=goal,
            prover=prover,
            informal_desc="rapid direct baseline without decomposition",
            context={"execution_mode": "rapid", "rapid_direct_path": "true"},
        )
    )
    return service.direct_match_result(
        goal,
        prover,
        match_result,
        execution_mode="rapid",
        informal_desc="Rapid-mode direct retrieval without architect decomposition.",
        context={"execution_mode": "rapid", "rapid_direct_path": "true"},
    )


async def _run_structured_single_pass(
    cdg: Any,
    *,
    prover: Prover,
    hunter: Any,
) -> OrchestratorResult:
    """Run one Hunter pass over decomposed leaves without orchestration refinement."""
    if _is_signal_event_rate_scaffold(cdg):
        match_results = _build_signal_event_rate_match_results(cdg, prover)
        return OrchestratorResult(
            cdg=cdg,
            match_results=match_results,
            rounds_used=1,
            failures=[],
            ungroundable=[],
        )

    pdg_nodes = to_pdg_nodes(cdg, prover=prover, strict=False)
    service = HunterService(hunter)
    batch_result = await service.match_batch(
        HunterBatchMatchRequest(pdg_nodes=pdg_nodes)
    )
    return OrchestratorResult(
        cdg=cdg,
        match_results=batch_result.match_results,
        rounds_used=1,
        failures=batch_result.failures,
        ungroundable=batch_result.ungroundable,
    )

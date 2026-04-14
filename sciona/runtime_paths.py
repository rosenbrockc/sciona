"""Shared runtime helpers for direct and structured execution paths."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from sciona.architect.handoff import CDGExport, to_pdg_nodes
from sciona.orchestrator import OrchestratorResult
from sciona.provider_expansion_declarations import (
    load_provider_expansion_declarations,
)
from sciona.services.hunter_service import (
    HunterBatchMatchRequest,
    HunterDirectMatchRequest,
    HunterService,
    build_direct_goal_cdg,
)
from sciona.services.models import MacroMatchRequest
from sciona.types import (
    Prover,
)


@lru_cache(maxsize=1)
def _signal_event_rate_declarations() -> dict[str, tuple[str, str, str]]:
    """Load signal-event-rate declarations from provider repos only."""
    return load_provider_expansion_declarations(
        "signal_event_rate",
        "SIGNAL_EVENT_RATE_DECLARATIONS",
    )


def _declaration_source_lib(
    declaration_name: str,
    *,
    fallback: str,
) -> str:
    """Infer a declaration module path from a dotted declaration name."""
    text = str(declaration_name or "").strip()
    if "." not in text:
        return fallback
    return text.rsplit(".", 1)[0]

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


async def _run_rapid_macro_match(
    goal: str,
    *,
    prover: Prover,
    artifact_retriever: Any,
) -> OrchestratorResult:
    """Run the rapid-mode direct macro retrieval path."""
    service = HunterService(None)
    match_result = await artifact_retriever.match_goal(MacroMatchRequest(goal=goal))
    return service.macro_match_result(
        goal,
        prover,
        match_result,
        execution_mode="rapid",
        informal_desc="Rapid-mode direct macro retrieval without architect decomposition.",
        context={"execution_mode": "rapid", "rapid_macro_direct_path": "true"},
    )


async def _run_structured_single_pass(
    cdg: Any,
    *,
    prover: Prover,
    hunter: Any,
) -> OrchestratorResult:
    """Run one Hunter pass over decomposed leaves without orchestration refinement."""
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

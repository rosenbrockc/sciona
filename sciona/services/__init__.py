"""Service-layer entrypoints for tool-orchestrated runtime modes."""

from __future__ import annotations

import importlib
from typing import Any

from sciona.services.models import (
    ArchitectDecomposeRequest,
    ArchitectDecomposeResult,
    HunterBatchMatchRequest,
    HunterBatchMatchResult,
    HunterDirectMatchRequest,
    MacroArtifactCandidate,
    MacroMatchRequest,
    MacroMatchResult,
    OrchestrationRequest,
    PlannerBudget,
    PlannerPolicy,
    PlannerRunResult,
    PlannerState,
    PlannerStep,
    SynthesizerAssembleAndCheckRequest,
    SynthesizerAssembleAndCheckResult,
    SynthesizerAssembleRequest,
    SynthesizerAssembleResult,
    SynthesizerCompileRequest,
    SynthesizerCompileResult,
    SynthesizerRepairRequest,
    SynthesizerRepairResult,
)

__all__ = [
    "ArchitectDecomposeRequest",
    "ArchitectDecomposeResult",
    "ArchitectService",
    "HunterBatchMatchRequest",
    "HunterBatchMatchResult",
    "HunterDirectMatchRequest",
    "HunterService",
    "MacroArtifactCandidate",
    "MacroMatchRequest",
    "MacroMatchResult",
    "OrchestrationRequest",
    "PlannerBudget",
    "PlannerPolicy",
    "OrchestratorService",
    "PlannerRunResult",
    "PlannerState",
    "PlannerStep",
    "SingleAgentPlanner",
    "SynthesizerAssembleAndCheckRequest",
    "SynthesizerAssembleAndCheckResult",
    "SynthesizerAssembleRequest",
    "SynthesizerAssembleResult",
    "SynthesizerCompileRequest",
    "SynthesizerCompileResult",
    "SynthesizerRepairRequest",
    "SynthesizerRepairResult",
    "SynthesizerService",
    "build_direct_goal_cdg",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "ArchitectService": ("sciona.services.architect_service", "ArchitectService"),
    "HunterService": ("sciona.services.hunter_service", "HunterService"),
    "build_direct_goal_cdg": (
        "sciona.services.hunter_service",
        "build_direct_goal_cdg",
    ),
    "OrchestratorService": (
        "sciona.services.orchestrator_service",
        "OrchestratorService",
    ),
    "SingleAgentPlanner": ("sciona.services.planner_service", "SingleAgentPlanner"),
    "SynthesizerService": ("sciona.services.synthesizer_service", "SynthesizerService"),
}


def __getattr__(name: str) -> Any:
    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attr_name = _LAZY_EXPORTS[name]
    module = importlib.import_module(module_name)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))

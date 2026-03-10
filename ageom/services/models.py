"""Shared request/response models for service-layer runtime wrappers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ageom.orchestrator import OrchestratorResult
from ageom.types import MatchFailureReport, MatchResult, PDGNode, Prover


@dataclass(frozen=True)
class ArchitectDecomposeRequest:
    """Request model for goal decomposition."""

    goal: str
    thread_id: str | None = None


@dataclass(frozen=True)
class ArchitectDecomposeResult:
    """Structured response from the architect service."""

    goal: str
    cdg: Any


@dataclass(frozen=True)
class HunterDirectMatchRequest:
    """Request model for direct goal matching without prior decomposition."""

    goal: str
    prover: Prover
    predicate_id: str = "goal_0"
    informal_desc: str = "single-agent direct grounding"
    context: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class HunterBatchMatchRequest:
    """Request model for matching an explicit batch of PDG nodes."""

    pdg_nodes: list[PDGNode]


@dataclass(frozen=True)
class HunterBatchMatchResult:
    """Batch match outputs with synthesized failure summaries."""

    match_results: list[MatchResult]
    failures: list[MatchFailureReport]
    ungroundable: list[str]


@dataclass(frozen=True)
class PlannerStep:
    """A planner-visible action taken during tool orchestration."""

    action: str
    detail: str
    status: str = "completed"


@dataclass(frozen=True)
class PlannerRunResult:
    """Top-level result for the first-cut single-agent planner runtime."""

    result: OrchestratorResult
    execution_path: str
    steps: list[PlannerStep]


@dataclass(frozen=True)
class OrchestrationRequest:
    """Request model for full orchestration escalation."""

    cdg: Any
    llm: Any
    prover: Prover
    max_rounds: int
    hunter_concurrency: int

"""Data contracts for the Principal (NAS / AutoML meta-optimizer) role."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class OptimizationMetric(str, Enum):
    """Metric axis that the Principal can optimise for."""

    LATENCY = "latency"
    MEMORY = "memory"
    PRECISION = "precision"
    FLOP_COUNT = "flop_count"
    STRUCTURE = "structure"
    CONVERGENCE = "convergence"


class NodeTelemetry(BaseModel, frozen=True):
    """Empirical metrics collected for a single CDG node during a benchmark run."""

    node_id: str = Field(..., description="CDG node identifier.")
    execution_time_ms: float = Field(
        ..., description="Wall-clock execution time in milliseconds."
    )
    peak_memory_bytes: int = Field(..., description="Peak resident memory in bytes.")
    error_expansion: float = Field(
        ...,
        description="Numerical error expansion factor relative to reference.",
    )


class BenchmarkResult(BaseModel, frozen=True):
    """Aggregate output of a single benchmark run over a synthesised pipeline."""

    global_loss: float = Field(
        ..., description="Scalar loss combining all active metrics."
    )
    node_telemetry: dict[str, NodeTelemetry] = Field(
        default_factory=dict,
        description="Per-node telemetry keyed by node_id.",
    )
    runtime_artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Best-effort runtime payloads captured during evaluation, such as "
            "stdout JSON, dataset-derived signal context, and explicit "
            "intermediate summaries."
        ),
    )


class ProposalStructuralDelta(BaseModel, frozen=True):
    """Compact structural delta from a baseline CDG to a proposal candidate."""

    node_count_delta: int = 0
    edge_count_delta: int = 0


class ProposalCandidateTrace(BaseModel):
    """Serializable trace of one evaluated refinement proposal."""

    label: str
    proposal_type: str
    candidate_type: str = ""
    loss: float | None = None
    improves_baseline: bool = False
    admissibility: dict[str, Any] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
    structural_delta: ProposalStructuralDelta = Field(
        default_factory=ProposalStructuralDelta
    )
    selected: bool = False
    selected_reason_codes: list[str] = Field(default_factory=list)
    rejected_reason_codes: list[str] = Field(default_factory=list)
    rules_applied: list[str] = Field(default_factory=list)
    applied_assets: list[dict[str, Any]] = Field(default_factory=list)
    variant_name: str = ""
    family: str = ""
    thread_id: str = ""
    diagnostic_count: int = 0
    diagnostic_rule_names: list[str] = Field(default_factory=list)
    context_summary: dict[str, Any] = Field(default_factory=dict)
    selection_disposition: str = ""
    selection_reason: str = ""


class ProposalSelectionTrace(BaseModel):
    """Serializable proposal-selection record for one trial."""

    baseline_loss: float
    candidates: list[ProposalCandidateTrace] = Field(default_factory=list)
    selected: str = ""
    selected_reason: str = ""
    selected_reason_codes: list[str] = Field(default_factory=list)
    skipped_due_to_admissibility: bool = False
    skip_reason: str = ""
    hard_reject_rule_ids: list[str] = Field(default_factory=list)


class NodeGradient(BaseModel, frozen=True):
    """Algorithmic partial derivative indicating optimisation pressure on a node."""

    node_id: str = Field(..., description="CDG node targeted by this gradient.")
    gradient_score: float = Field(
        ...,
        description="Magnitude of the optimisation signal (higher = more pressure).",
    )
    metric_type: OptimizationMetric = Field(
        ..., description="Which metric axis this gradient targets."
    )
    bottleneck_reason: str = Field(
        ...,
        description=(
            "Natural-language explanation of why the node is a bottleneck, "
            "e.g. \"Node 'sort_array' consumed 85%% of total execution time\"."
        ),
    )

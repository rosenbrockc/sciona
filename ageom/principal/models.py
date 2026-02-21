"""Data contracts for the Principal (NAS / AutoML meta-optimizer) role."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class OptimizationMetric(str, Enum):
    """Metric axis that the Principal can optimise for."""

    LATENCY = "latency"
    MEMORY = "memory"
    PRECISION = "precision"
    FLOP_COUNT = "flop_count"


class NodeTelemetry(BaseModel, frozen=True):
    """Empirical metrics collected for a single CDG node during a benchmark run."""

    node_id: str = Field(..., description="CDG node identifier.")
    execution_time_ms: float = Field(
        ..., description="Wall-clock execution time in milliseconds."
    )
    peak_memory_bytes: int = Field(
        ..., description="Peak resident memory in bytes."
    )
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

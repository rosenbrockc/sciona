"""Structural objective helpers derived from ghost simulation."""

from __future__ import annotations

from sciona.principal.models import BenchmarkResult
from sciona.synthesizer.ghost_sim import GhostSimReport


def benchmark_from_ghost_report(report: GhostSimReport) -> BenchmarkResult:
    """Convert a ghost simulation report into a benchmark-style loss."""
    return BenchmarkResult(global_loss=compute_structure_loss(report))


def compute_structure_loss(report: GhostSimReport) -> float:
    """Lower is better: 0 means fully simulable and structurally sound."""
    if not report.ran:
        return 1.0

    loss = max(0.0, 1.0 - float(report.coverage))
    if not report.passed:
        loss += 1.0
    if report.cyclic_deadlock:
        loss += 0.5
    if report.error_node:
        loss += 0.25
    return loss

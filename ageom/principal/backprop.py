"""Credit assignment: compute per-node optimisation gradients from telemetry."""

from __future__ import annotations


from ageom.architect.handoff import CDGExport
from ageom.architect.models import NodeStatus
from ageom.principal.models import (
    BenchmarkResult,
    NodeGradient,
    OptimizationMetric,
)
from ageom.synthesizer.ghost_sim import GhostSimReport


class CreditAssigner:
    """Deterministic credit assignment from empirical telemetry to CDG nodes."""

    def compute_gradients(
        self,
        cdg: CDGExport,
        benchmark: BenchmarkResult,
        sim_report: GhostSimReport,
        target: OptimizationMetric,
    ) -> list[NodeGradient]:
        """Compute per-node optimisation gradients.

        Args:
            cdg: The Conceptual Dependency Graph.
            benchmark: Empirical telemetry from a benchmark run.
            sim_report: Ghost simulation report (carries precision gradients).
            target: Which metric axis to compute gradients for.

        Returns:
            ``NodeGradient`` objects sorted descending by ``gradient_score``.
        """
        atomic_ids = {n.node_id for n in cdg.nodes if n.status == NodeStatus.ATOMIC}
        node_names = {n.node_id: n.name for n in cdg.nodes}

        if target in (OptimizationMetric.LATENCY, OptimizationMetric.FLOP_COUNT):
            return self._gradient_latency(
                atomic_ids,
                node_names,
                benchmark,
                target,
            )
        if target == OptimizationMetric.MEMORY:
            return self._gradient_memory(
                atomic_ids,
                node_names,
                benchmark,
            )
        # PRECISION
        return self._gradient_precision(
            atomic_ids,
            node_names,
            benchmark,
            sim_report,
        )

    # ------------------------------------------------------------------
    # Metric-specific helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _gradient_latency(
        atomic_ids: set[str],
        node_names: dict[str, str],
        benchmark: BenchmarkResult,
        target: OptimizationMetric,
    ) -> list[NodeGradient]:
        total_ms = sum(
            t.execution_time_ms
            for nid, t in benchmark.node_telemetry.items()
            if nid in atomic_ids
        )
        if total_ms <= 0:
            return []

        gradients: list[NodeGradient] = []
        for nid in atomic_ids:
            tel = benchmark.node_telemetry.get(nid)
            if tel is None:
                continue
            pct = tel.execution_time_ms / total_ms * 100.0
            gradients.append(
                NodeGradient(
                    node_id=nid,
                    gradient_score=pct,
                    metric_type=target,
                    bottleneck_reason=(
                        f"Node '{node_names.get(nid, nid)}' consumed "
                        f"{pct:.1f}% of total execution time"
                    ),
                )
            )

        gradients.sort(key=lambda g: g.gradient_score, reverse=True)
        return gradients

    @staticmethod
    def _gradient_memory(
        atomic_ids: set[str],
        node_names: dict[str, str],
        benchmark: BenchmarkResult,
    ) -> list[NodeGradient]:
        total_bytes = sum(
            t.peak_memory_bytes
            for nid, t in benchmark.node_telemetry.items()
            if nid in atomic_ids
        )
        if total_bytes <= 0:
            return []

        gradients: list[NodeGradient] = []
        for nid in atomic_ids:
            tel = benchmark.node_telemetry.get(nid)
            if tel is None:
                continue
            pct = tel.peak_memory_bytes / total_bytes * 100.0
            gradients.append(
                NodeGradient(
                    node_id=nid,
                    gradient_score=pct,
                    metric_type=OptimizationMetric.MEMORY,
                    bottleneck_reason=(
                        f"Node '{node_names.get(nid, nid)}' consumed "
                        f"{pct:.1f}% of total peak memory"
                    ),
                )
            )

        gradients.sort(key=lambda g: g.gradient_score, reverse=True)
        return gradients

    @staticmethod
    def _gradient_precision(
        atomic_ids: set[str],
        node_names: dict[str, str],
        benchmark: BenchmarkResult,
        sim_report: GhostSimReport,
    ) -> list[NodeGradient]:
        gradients: list[NodeGradient] = []

        # Combine empirical error_expansion with ghost-sim interval data.
        # Ghost-sim precision_gradients are the primary signal; telemetry
        # error_expansion is the fallback.
        scored: dict[str, float] = {}
        for nid in atomic_ids:
            pg = sim_report.precision_gradients.get(nid)
            if pg is not None and pg != 0.0:
                scored[nid] = abs(pg)
                continue
            tel = benchmark.node_telemetry.get(nid)
            if tel is not None and tel.error_expansion > 0:
                scored[nid] = tel.error_expansion

        total = sum(scored.values())
        if total <= 0:
            return []

        for nid, raw in scored.items():
            pct = raw / total * 100.0
            gradients.append(
                NodeGradient(
                    node_id=nid,
                    gradient_score=pct,
                    metric_type=OptimizationMetric.PRECISION,
                    bottleneck_reason=(
                        f"Node '{node_names.get(nid, nid)}' contributed "
                        f"{pct:.1f}% of total numerical error expansion"
                    ),
                )
            )

        gradients.sort(key=lambda g: g.gradient_score, reverse=True)
        return gradients

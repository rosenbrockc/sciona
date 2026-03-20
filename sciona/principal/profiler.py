"""Standalone error profiler for evaluating existing CDGs and compiled artifacts."""

from __future__ import annotations

import logging

from sciona.architect.handoff import CDGExport
from sciona.principal.backprop import CreditAssigner
from sciona.principal.evaluator import ExecutionSandbox
from sciona.principal.models import NodeGradient, OptimizationMetric
from sciona.principal.reference_attribution import (
    compute_reference_loss_gradients,
    is_reference_loss_objective,
)
from sciona.principal.structure_objective import benchmark_from_ghost_report
from sciona.synthesizer.ghost_sim import run_ghost_simulation
from sciona.synthesizer.models import ExportBundle
from sciona.types import MatchResult

logger = logging.getLogger(__name__)


async def profile_algorithm_error(
    cdg: CDGExport,
    bundle: ExportBundle,
    dataset_path: str,
    metric: OptimizationMetric = OptimizationMetric.PRECISION,
    *,
    dataset_varset: dict[str, str] | None = None,
    match_results: list[MatchResult] | None = None,
    evaluation_spec: dict | str | None = None,
) -> list[NodeGradient]:
    """Evaluate an existing CDG against a dataset and rank error contributors.

    Args:
        cdg: The completed Conceptual Dependency Graph.
        bundle: The compiled artifact export bundle to evaluate.
        dataset_path: Path to the benchmark dataset.
        metric: The optimization metric to profile (defaults to PRECISION).

    Returns:
        A list of NodeGradient objects ranked by their contribution to the loss.
    """
    logger.info("Profiling artifact %s against dataset %s", bundle.compiled_artifact or bundle.source_path, dataset_path)
    
    ghost_report = run_ghost_simulation(cdg, match_results=match_results or [])

    if metric == OptimizationMetric.STRUCTURE:
        benchmark = benchmark_from_ghost_report(ghost_report)
    else:
        if is_reference_loss_objective(metric, evaluation_spec)[0] is not None:
            gradients = await compute_reference_loss_gradients(
                cdg,
                bundle,
                dataset_path,
                evaluation_spec,
                dataset_varset=dataset_varset,
            )
            if gradients:
                return gradients
        sandbox = ExecutionSandbox()
        if dataset_path.endswith((".yml", ".yaml")):
            benchmark = await sandbox.evaluate_adapter(
                bundle,
                dataset_path,
                metric,
                varset=dataset_varset,
                evaluation_spec=evaluation_spec,
            )
        else:
            benchmark = await sandbox.evaluate(
                bundle,
                dataset_path,
                metric,
                evaluation_spec=evaluation_spec,
            )

    assigner = CreditAssigner()
    gradients = assigner.compute_gradients(cdg, benchmark, ghost_report, metric)

    return gradients

"""Standalone error profiler for evaluating existing CDGs and compiled artifacts."""

from __future__ import annotations

import logging

from ageom.architect.handoff import CDGExport
from ageom.principal.backprop import CreditAssigner
from ageom.principal.evaluator import ExecutionSandbox
from ageom.principal.models import NodeGradient, OptimizationMetric
from ageom.synthesizer.ghost_sim import run_ghost_simulation
from ageom.synthesizer.models import ExportBundle

logger = logging.getLogger(__name__)


async def profile_algorithm_error(
    cdg: CDGExport,
    bundle: ExportBundle,
    dataset_path: str,
    metric: OptimizationMetric = OptimizationMetric.PRECISION,
    *,
    dataset_varset: dict[str, str] | None = None,
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
    
    # 1. Execute the compiled artifact against the dataset to gather empirical telemetry
    sandbox = ExecutionSandbox()
    if dataset_path.endswith((".yml", ".yaml")):
        benchmark = await sandbox.evaluate_adapter(
            bundle,
            dataset_path,
            metric,
            varset=dataset_varset,
        )
    else:
        benchmark = await sandbox.evaluate(bundle, dataset_path, metric)
    
    # 2. Run the ghost simulation for structural checks and theoretical precision bounds
    # Assuming empty match_results as we just want theoretical bounds for the CDG nodes.
    ghost_report = run_ghost_simulation(cdg, match_results=[])
    
    # 3. Assign credit / compute gradients
    assigner = CreditAssigner()
    gradients = assigner.compute_gradients(cdg, benchmark, ghost_report, metric)
    
    return gradients

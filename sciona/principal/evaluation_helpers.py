"""Shared execution helpers for Principal evaluation paths."""

from __future__ import annotations

import inspect
from typing import Any

from sciona.principal.models import BenchmarkResult, OptimizationMetric
from sciona.synthesizer.models import ExportBundle


def _accepted_kwargs(callable_obj: Any, candidates: dict[str, Any]) -> dict[str, Any]:
    """Filter keyword arguments to those accepted by *callable_obj*."""
    try:
        signature = inspect.signature(callable_obj)
    except (TypeError, ValueError):
        return candidates

    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return candidates

    return {
        name: value
        for name, value in candidates.items()
        if name in signature.parameters
    }


async def evaluate_bundle_for_metric(
    sandbox: Any,
    bundle: ExportBundle,
    dataset_path: str,
    metric: OptimizationMetric,
    *,
    dataset_varset: dict[str, str] | None = None,
    evaluation_spec: dict[str, Any] | str | None = None,
) -> BenchmarkResult:
    """Evaluate a bundle, adapting to older sandbox adapter signatures."""
    if dataset_path.endswith((".yml", ".yaml")):
        kwargs = _accepted_kwargs(
            sandbox.evaluate_adapter,
            {
                "varset": dataset_varset,
                "evaluation_spec": evaluation_spec,
            },
        )
        return await sandbox.evaluate_adapter(bundle, dataset_path, metric, **kwargs)

    kwargs = _accepted_kwargs(
        sandbox.evaluate,
        {
            "evaluation_spec": evaluation_spec,
        },
    )
    return await sandbox.evaluate(bundle, dataset_path, metric, **kwargs)

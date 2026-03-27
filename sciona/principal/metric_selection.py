"""Helpers for resolving user-facing optimization objectives."""

from __future__ import annotations

from typing import Any

from sciona.principal.eval_spec import load_evaluation_spec
from sciona.principal.models import OptimizationMetric

SUPPORTED_OBJECTIVES: tuple[str, ...] = (
    "latency",
    "memory",
    "precision",
    "uncertainty",
    "rmse",
    "mse",
    "mae",
    "flop_count",
    "structure",
    "convergence",
)


def resolve_optimization_objective(
    objective: str,
    evaluation_spec: dict[str, Any] | str | None = None,
) -> tuple[OptimizationMetric, dict[str, Any] | str | None, str]:
    """Resolve a user-facing objective string into the internal metric."""
    normalized = str(objective).strip().lower()
    if normalized == "uncertainty":
        return OptimizationMetric.PRECISION, evaluation_spec, normalized
    if normalized in {"rmse", "mse", "mae"}:
        if evaluation_spec is None:
            raise ValueError(
                f"objective '{normalized}' requires an evaluation spec with reference alignment"
            )
        spec_payload = _coerce_eval_spec_dict(evaluation_spec)
        spec_payload["loss"] = normalized
        return OptimizationMetric.PRECISION, spec_payload, normalized
    if normalized == OptimizationMetric.STRUCTURE.value:
        return OptimizationMetric.STRUCTURE, evaluation_spec, normalized
    try:
        return OptimizationMetric(normalized), evaluation_spec, normalized
    except ValueError as exc:
        supported = ", ".join(SUPPORTED_OBJECTIVES)
        raise ValueError(
            f"unsupported objective '{objective}'; expected one of: {supported}"
        ) from exc


def _coerce_eval_spec_dict(
    evaluation_spec: dict[str, Any] | str,
) -> dict[str, Any]:
    if isinstance(evaluation_spec, dict):
        return dict(evaluation_spec)
    payload = load_evaluation_spec(evaluation_spec)
    if payload is None:
        raise ValueError("evaluation spec could not be loaded")
    return dict(payload)

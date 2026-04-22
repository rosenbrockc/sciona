"""Derive first-class runtime heuristics from canonical telemetry summaries."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from pydantic import BaseModel, Field

from sciona.heuristics import (
    CanonicalHeuristic,
    HeuristicActionClass,
    HeuristicApplicabilityScope,
    HeuristicEvidenceType,
    HeuristicProducerKind,
)


class RuntimeHeuristicObservation(BaseModel):
    """One heuristic observation derived from canonical runtime telemetry."""

    heuristic: CanonicalHeuristic
    source_section: str
    source_key: str = ""
    summary_path: str = ""
    metric_name: str = ""
    metric_value: float | None = None
    threshold: float | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    provenance: str = "canonical_telemetry_summary"
    supporting_fields: dict[str, Any] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)


class RuntimeHeuristicEvidence(BaseModel):
    """Compact runtime heuristic bundle persisted alongside telemetry."""

    observations: list[RuntimeHeuristicObservation] = Field(default_factory=list)
    heuristic_summary: dict[str, Any] = Field(default_factory=dict)


_MAX_OUTLIER_FRACTION = 0.05
_MIN_DENSITY_PER_MINUTE = 5.0
_MIN_DISPERSION = 0.5


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def _heuristic(
    heuristic_id: str,
    *,
    display_name: str,
    meaning: str,
    evidence_type: HeuristicEvidenceType,
    action_classes: list[HeuristicActionClass],
    confidence: float,
) -> CanonicalHeuristic:
    return CanonicalHeuristic(
        heuristic_id=heuristic_id,
        display_name=display_name,
        dejargonized_meaning=meaning,
        evidence_type=evidence_type,
        value_kind="scalar_score",
        value_shape="scalar",
        confidence=_clamp01(confidence),
        producer_kind=HeuristicProducerKind.RUNTIME_TRANSFORM,
        applicability_scope=HeuristicApplicabilityScope.CROSS_FAMILY,
        supported_action_classes=action_classes,
        provenance_requirements=["source_section", "metric_name", "metric_value"],
    )


def _observation(
    *,
    heuristic_id: str,
    display_name: str,
    meaning: str,
    evidence_type: HeuristicEvidenceType,
    action_classes: list[HeuristicActionClass],
    source_section: str,
    metric_name: str,
    metric_value: float,
    confidence: float,
    threshold: float | None = None,
    source_key: str = "",
    summary_path: str = "",
    notes: list[str] | None = None,
    supporting_fields: dict[str, Any] | None = None,
) -> RuntimeHeuristicObservation:
    return RuntimeHeuristicObservation(
        heuristic=_heuristic(
            heuristic_id,
            display_name=display_name,
            meaning=meaning,
            evidence_type=evidence_type,
            action_classes=action_classes,
            confidence=confidence,
        ),
        source_section=source_section,
        source_key=source_key,
        summary_path=summary_path or f"{source_section}.{metric_name}",
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        confidence=_clamp01(confidence),
        notes=list(notes or []),
        supporting_fields=dict(supporting_fields or {}),
    )


def _summary_value(summary: Mapping[str, Any], key: str) -> float | None:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _quality_dispersion(summary: Mapping[str, Any]) -> float | None:
    std = _summary_value(summary, "std")
    if std is None:
        return None
    mean = abs(_summary_value(summary, "mean") or 0.0)
    p50 = abs(_summary_value(summary, "p50") or 0.0)
    scale = max(mean, p50, 1.0)
    return std / scale


def _derive_from_summary(
    *,
    source_section: str,
    summary: Mapping[str, Any],
    source_key: str = "",
    summary_path: str = "",
) -> list[RuntimeHeuristicObservation]:
    observations: list[RuntimeHeuristicObservation] = []
    count = _summary_value(summary, "count") or 0.0
    discontinuity_count = _summary_value(summary, "discontinuity_count") or 0.0
    outlier_fraction = _summary_value(summary, "outlier_fraction") or 0.0
    density_per_minute = _summary_value(summary, "density_per_minute")
    dispersion = _quality_dispersion(summary)

    if discontinuity_count > 0:
        confidence = min(1.0, 0.55 + min(discontinuity_count, 5.0) * 0.1)
        observations.append(
            _observation(
                heuristic_id="boundary_discontinuity",
                display_name="Boundary Discontinuity",
                meaning=(
                    "The observed boundary contains abrupt discontinuities that may "
                    "need cleanup before downstream processing."
                ),
                evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
                action_classes=[
                    HeuristicActionClass.PRECONDITION,
                    HeuristicActionClass.INSERT_CORRECTION,
                ],
                source_section=source_section,
                source_key=source_key,
                summary_path=summary_path,
                metric_name="discontinuity_count",
                metric_value=discontinuity_count,
                confidence=confidence,
                threshold=0.0,
                notes=[
                    "Derived from canonical telemetry_summary discontinuity_count.",
                ],
                supporting_fields=dict(summary),
            )
        )

    if outlier_fraction > _MAX_OUTLIER_FRACTION:
        confidence = min(1.0, 0.5 + min(outlier_fraction, 1.0) * 0.5)
        observations.append(
            _observation(
                heuristic_id="interval_instability",
                display_name="Interval Instability",
                meaning=(
                    "Observed intervals vary enough that the downstream summary may "
                    "benefit from smoothing or corrective handling."
                ),
                evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
                action_classes=[
                    HeuristicActionClass.SMOOTH_OR_AGGREGATE,
                    HeuristicActionClass.INSERT_CORRECTION,
                ],
                source_section=source_section,
                source_key=source_key,
                summary_path=summary_path,
                metric_name="outlier_fraction",
                metric_value=outlier_fraction,
                confidence=confidence,
                threshold=_MAX_OUTLIER_FRACTION,
                notes=[
                    "Derived from canonical telemetry_summary outlier_fraction.",
                ],
                supporting_fields=dict(summary),
            )
        )

    if density_per_minute is not None and count > 0 and density_per_minute < _MIN_DENSITY_PER_MINUTE:
        confidence = min(1.0, 0.5 + (_MIN_DENSITY_PER_MINUTE - density_per_minute) / _MIN_DENSITY_PER_MINUTE)
        observations.append(
            _observation(
                heuristic_id="density_collapse",
                display_name="Density Collapse",
                meaning=(
                    "The observed structure is too sparse to support reliable "
                    "downstream decisions."
                ),
                evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
                action_classes=[
                    HeuristicActionClass.GATE_OR_VALIDATE,
                    HeuristicActionClass.BRANCH_AND_COMPARE,
                ],
                source_section=source_section,
                source_key=source_key,
                summary_path=summary_path,
                metric_name="density_per_minute",
                metric_value=density_per_minute,
                confidence=confidence,
                threshold=_MIN_DENSITY_PER_MINUTE,
                notes=[
                    "Derived from canonical telemetry_summary density_per_minute.",
                ],
                supporting_fields=dict(summary),
            )
        )

    if dispersion is not None and dispersion >= _MIN_DISPERSION:
        confidence = min(1.0, 0.45 + min(dispersion, 2.0) * 0.25)
        observations.append(
            _observation(
                heuristic_id="quality_instability",
                display_name="Quality Instability",
                meaning=(
                    "The observed summary varies enough that a fixed downstream "
                    "path may be brittle."
                ),
                evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
                action_classes=[
                    HeuristicActionClass.GATE_OR_VALIDATE,
                    HeuristicActionClass.BRANCH_AND_COMPARE,
                ],
                source_section=source_section,
                source_key=source_key,
                summary_path=summary_path,
                metric_name="dispersion",
                metric_value=dispersion,
                confidence=confidence,
                threshold=_MIN_DISPERSION,
                notes=[
                    "Derived from dispersion over canonical telemetry summary fields.",
                ],
                supporting_fields=dict(summary),
            )
        )

    return observations


_MODEL_SELECTION_DIAGNOSTICS: list[
    tuple[str, str, str, str, float, list[HeuristicActionClass]]
] = [
    # (intermediates_key, heuristic_id, display_name, meaning, threshold, action_classes)
    (
        "condition_number",
        "numerical_condition_instability",
        "Numerical Condition Instability",
        "Design matrix ill-conditioning degrades estimator stability.",
        30.0,
        [HeuristicActionClass.PRECONDITION, HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "n_p_ratio",
        "coverage_fragmentation",
        "Sample Coverage Fragmentation",
        "Insufficient samples relative to features requires regularization.",
        10.0,
        [HeuristicActionClass.PRECONDITION, HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "mutual_incoherence",
        "constraint_violation_risk",
        "Feature Incoherence Risk",
        "High mutual incoherence breaks Lasso recovery guarantees.",
        1.0,
        [HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "excess_kurtosis",
        "residual_structure_after_transform",
        "Target Kurtosis",
        "Heavy-tailed target suggests robust loss function.",
        1.0,
        [HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "residual_kurtosis",
        "residual_structure_after_transform",
        "Residual Kurtosis",
        "Heavy-tailed residuals suggest robust loss function.",
        1.0,
        [HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "noise_level",
        "quality_instability",
        "Noise Level",
        "High noise favors bagging (RandomForest) over boosting.",
        0.5,
        [HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "dispersion_index",
        "density_collapse",
        "Dispersion Index",
        "Mean-variance relationship deviates from Gaussian assumption.",
        1.1,
        [HeuristicActionClass.REPLACE_STAGE],
    ),
    (
        "vif_max",
        "numerical_condition_instability",
        "Variance Inflation Factor",
        "Severe multicollinearity requires PCA or feature removal.",
        10.0,
        [HeuristicActionClass.PRECONDITION, HeuristicActionClass.INSERT_CORRECTION],
    ),
    (
        "skewness_max",
        "residual_structure_after_transform",
        "Feature Skewness",
        "Highly skewed features need power transform.",
        1.0,
        [HeuristicActionClass.PRECONDITION],
    ),
    (
        "explained_variance",
        "coverage_fragmentation",
        "Explained Variance",
        "Low explained variance indicates need for dimensionality reduction.",
        0.95,
        [HeuristicActionClass.PRECONDITION],
    ),
]

_MODEL_SELECTION_PREFIX = "model_selection"


def _derive_from_model_selection_intermediates(
    intermediates: Mapping[str, Any],
) -> list[RuntimeHeuristicObservation]:
    """Derive heuristic observations from model-selection diagnostic atom outputs."""
    observations: list[RuntimeHeuristicObservation] = []

    for (
        key, heuristic_id, display_name, meaning, threshold, action_classes,
    ) in _MODEL_SELECTION_DIAGNOSTICS:
        value = intermediates.get(f"{_MODEL_SELECTION_PREFIX}.{key}")
        if value is None:
            continue
        try:
            metric_value = float(value)
        except (ValueError, TypeError):
            continue
        # For explained_variance, trigger when BELOW threshold; others when above
        if key == "explained_variance":
            if metric_value >= threshold:
                continue
            confidence = _clamp01(0.6 + (threshold - metric_value) * 2.0)
        else:
            if metric_value <= threshold:
                continue
            gap = (metric_value - threshold) / max(threshold, 1.0)
            confidence = _clamp01(0.6 + min(gap, 1.0) * 0.35)

        observations.append(
            _observation(
                heuristic_id=heuristic_id,
                display_name=display_name,
                meaning=meaning,
                evidence_type=HeuristicEvidenceType.SCALAR_SCORE,
                action_classes=action_classes,
                source_section=_MODEL_SELECTION_PREFIX,
                metric_name=key,
                metric_value=metric_value,
                confidence=confidence,
                threshold=threshold,
                notes=[f"Derived from model selection diagnostic atom output {key}."],
            )
        )

    # Boolean diagnostics
    is_ts = intermediates.get(f"{_MODEL_SELECTION_PREFIX}.is_time_series")
    if is_ts is True or is_ts == 1:
        observations.append(
            _observation(
                heuristic_id="cv_strategy_selection",
                display_name="Time Series Detection",
                meaning="Temporal structure detected — TimeSeriesSplit mandatory.",
                evidence_type=HeuristicEvidenceType.BOOLEAN_FLAG,
                action_classes=[HeuristicActionClass.GATE_OR_VALIDATE],
                source_section=_MODEL_SELECTION_PREFIX,
                metric_name="is_time_series",
                metric_value=1.0,
                confidence=0.95,
                threshold=0.0,
                notes=["First column is strictly monotonically increasing."],
            )
        )

    is_sparse = intermediates.get(f"{_MODEL_SELECTION_PREFIX}.is_sparse")
    if is_sparse is True or is_sparse == 1:
        observations.append(
            _observation(
                heuristic_id="preprocessing_selection",
                display_name="Sparse Matrix Detection",
                meaning="Sparse input requires sparse-safe preprocessing and solvers.",
                evidence_type=HeuristicEvidenceType.BOOLEAN_FLAG,
                action_classes=[
                    HeuristicActionClass.PRECONDITION,
                    HeuristicActionClass.REPLACE_STAGE,
                ],
                source_section=_MODEL_SELECTION_PREFIX,
                metric_name="is_sparse",
                metric_value=1.0,
                confidence=0.90,
                threshold=0.0,
                notes=["Input X is a scipy sparse matrix."],
            )
        )

    return observations


def derive_runtime_heuristics(
    runtime_evidence: Mapping[str, Any] | None,
    intermediates: Mapping[str, Any] | None = None,
) -> RuntimeHeuristicEvidence:
    """Derive generic heuristics from canonical runtime evidence summaries.

    Parameters
    ----------
    runtime_evidence
        Canonical telemetry summaries (signal, events, rate, etc.).
    intermediates
        Expansion context intermediates containing model-selection atom
        outputs under the ``model_selection.*`` prefix.
    """
    all_observations: list[RuntimeHeuristicObservation] = []

    # Model-selection observations from intermediates
    if isinstance(intermediates, Mapping):
        has_ms = any(
            k.startswith(f"{_MODEL_SELECTION_PREFIX}.")
            for k in intermediates
        )
        if has_ms:
            all_observations.extend(
                _derive_from_model_selection_intermediates(intermediates)
            )

    if not isinstance(runtime_evidence, Mapping):
        if all_observations:
            return _build_evidence(all_observations)
        return RuntimeHeuristicEvidence()

    telemetry_summary = runtime_evidence.get("telemetry_summary", runtime_evidence)
    if not isinstance(telemetry_summary, Mapping):
        return RuntimeHeuristicEvidence()

    observations: list[RuntimeHeuristicObservation] = []

    def extend_from(section: str, payload: Any, *, source_key: str = "") -> None:
        if isinstance(payload, Mapping):
            observations.extend(
                _derive_from_summary(
                    source_section=section,
                    summary=payload,
                    source_key=source_key,
                    summary_path=f"telemetry_summary.{section}",
                )
            )

    extend_from("signal", telemetry_summary.get("signal", {}), source_key=str(telemetry_summary.get("signal", {}).get("source_key", "") or ""))
    extend_from("events", telemetry_summary.get("events", {}), source_key=str(telemetry_summary.get("events", {}).get("source_key", "") or ""))
    extend_from("rate", telemetry_summary.get("rate", {}), source_key=str(telemetry_summary.get("rate", {}).get("source_key", "") or ""))

    intermediates = telemetry_summary.get("intermediates", {})
    if isinstance(intermediates, Mapping):
        for name, summary in intermediates.items():
            if name in {"events", "rate"}:
                continue
            extend_from(f"intermediates.{name}", summary, source_key=str(name))

    outputs = telemetry_summary.get("outputs", {})
    if isinstance(outputs, Mapping):
        for name, summary in outputs.items():
            extend_from(f"outputs.{name}", summary, source_key=str(name))

    all_observations.extend(observations)
    return _build_evidence(all_observations)


def _build_evidence(
    observations: list[RuntimeHeuristicObservation],
) -> RuntimeHeuristicEvidence:
    """Deduplicate, sort, and wrap observations into an evidence bundle."""
    deduped: dict[tuple[str, str, str], RuntimeHeuristicObservation] = {}
    for observation in observations:
        key = (
            observation.heuristic.heuristic_id,
            observation.source_section,
            observation.source_key,
        )
        deduped.setdefault(key, observation)

    ordered = sorted(
        deduped.values(),
        key=lambda item: (
            item.heuristic.heuristic_id,
            item.source_section,
            item.source_key,
        ),
    )
    summary: dict[str, Any] = {
        "heuristic_count": len(ordered),
        "heuristic_ids": [item.heuristic.heuristic_id for item in ordered],
        "source_sections": [item.source_section for item in ordered],
        "source_keys": [item.source_key for item in ordered],
        "max_confidence": max((item.confidence for item in ordered), default=0.0),
    }
    return RuntimeHeuristicEvidence(observations=ordered, heuristic_summary=summary)

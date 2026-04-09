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


def derive_runtime_heuristics(
    runtime_evidence: Mapping[str, Any] | None,
) -> RuntimeHeuristicEvidence:
    """Derive generic heuristics from canonical runtime evidence summaries."""
    if not isinstance(runtime_evidence, Mapping):
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

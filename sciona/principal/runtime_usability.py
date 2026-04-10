"""Runtime usability assessment emission from canonical evidence."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from sciona.usability import (
    UsabilityAssessment,
    UsabilityBlockingReasonCode,
    UsabilityProvenance,
    UsabilityProvenanceKind,
    UsabilityReason,
    UsabilityReasonKind,
    UsabilityScope,
    UsabilityScopeAssessment,
    UsabilityWarningReasonCode,
)


_ASSESSMENT_ID = "runtime_usability_assessment"
_TASK_INTENT = "runtime_artifact_emission"
_REQUIRED_CONTRACTS = (
    "runtime_context",
    "canonical_runtime_context",
    "telemetry_summary",
    "heuristics",
    "heuristic_summary",
)


def _provenance(source_id: str, *, note: str = "") -> UsabilityProvenance:
    return UsabilityProvenance(
        kind=UsabilityProvenanceKind.RUNTIME_ASSESSOR,
        source_id=source_id,
        note=note,
    )


def _reason(
    kind: UsabilityReasonKind,
    code: UsabilityBlockingReasonCode | UsabilityWarningReasonCode,
    *,
    summary: str,
    source_id: str,
    related_heuristics: list[str] | None = None,
    confidence: float = 0.75,
) -> UsabilityReason:
    uncertainty_notes = []
    if confidence < 1.0:
        uncertainty_notes.append(
            "Derived from runtime evidence and may require broader corroboration."
        )
    return UsabilityReason(
        kind=kind,
        code=code.value,
        summary=summary,
        related_heuristic_ids=list(related_heuristics or []),
        confidence=confidence,
        uncertainty_notes=uncertainty_notes,
        provenance=[_provenance(source_id, note=summary)],
    )


def _heuristic_ids(runtime_evidence: Mapping[str, Any]) -> list[str]:
    heuristic_ids: list[str] = []
    for item in runtime_evidence.get("heuristics", []) or []:
        if isinstance(item, Mapping):
            heuristic = item.get("heuristic", {})
            if isinstance(heuristic, Mapping):
                heuristic_id = str(heuristic.get("heuristic_id", "") or "").strip()
                if heuristic_id:
                    heuristic_ids.append(heuristic_id)
    summary = runtime_evidence.get("heuristic_summary", {})
    if isinstance(summary, Mapping):
        for item in summary.get("heuristic_ids", []) or []:
            heuristic_id = str(item or "").strip()
            if heuristic_id:
                heuristic_ids.append(heuristic_id)
    return sorted(dict.fromkeys(heuristic_ids))


def _confidence_from_support(heuristic_count: int, max_confidence: float) -> float:
    if heuristic_count <= 0:
        return 0.25
    return max(0.4, min(0.95, 0.45 + (max_confidence * 0.5)))


def _guidance_scope(
    *,
    runtime_context: Mapping[str, Any],
    telemetry_summary: Mapping[str, Any],
    heuristic_ids: list[str],
    heuristic_count: int,
    max_confidence: float,
) -> UsabilityScopeAssessment:
    blocking_reasons: list[UsabilityReason] = []
    warning_reasons: list[UsabilityReason] = []
    usable = bool(runtime_context)
    confidence = _confidence_from_support(heuristic_count, max_confidence)

    if not runtime_context:
        blocking_reasons.append(
            _reason(
                UsabilityReasonKind.BLOCKING,
                UsabilityBlockingReasonCode.REQUIRED_INPUT_MISSING,
                summary="Runtime context is missing, so guidance cannot be emitted reliably.",
                source_id="runtime_context",
                related_heuristics=heuristic_ids,
                confidence=0.95,
            )
        )
        usable = False
    if not telemetry_summary:
        blocking_reasons.append(
            _reason(
                UsabilityReasonKind.BLOCKING,
                UsabilityBlockingReasonCode.COVERAGE_INSUFFICIENT,
                summary="Telemetry summary is missing, so guidance coverage is incomplete.",
                source_id="telemetry_summary",
                related_heuristics=heuristic_ids,
                confidence=0.9,
            )
        )
        usable = False
    if heuristic_count == 0:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.PARTIAL_COVERAGE,
                summary="No runtime heuristics were derived from the evidence.",
                source_id="heuristic_summary",
                related_heuristics=[],
                confidence=0.7,
            )
        )
    else:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.REVIEW_RECOMMENDED,
                summary="Heuristic support is present but still benefits from review.",
                source_id="heuristic_summary",
                related_heuristics=heuristic_ids,
                confidence=min(0.9, max_confidence or 0.7),
            )
        )
    if heuristic_count <= 1:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.NARROW_SUPPORT,
                summary="The evidence relies on a narrow heuristic signature.",
                source_id="heuristic_summary",
                related_heuristics=heuristic_ids,
                confidence=0.65,
            )
        )

    return UsabilityScopeAssessment(
        scope=UsabilityScope.GUIDANCE,
        usable=usable,
        confidence=confidence,
        uncertainty_notes=[
            "Runtime guidance is derived only from canonical runtime evidence.",
        ],
        provenance=[_provenance("runtime_usability", note="guidance_scope")],
        blocking_reasons=blocking_reasons,
        warning_reasons=warning_reasons,
    )


def _scoring_scope(
    *,
    guidance_usable: bool,
    telemetry_summary: Mapping[str, Any],
    heuristic_ids: list[str],
    heuristic_count: int,
    max_confidence: float,
) -> UsabilityScopeAssessment:
    blocking_reasons: list[UsabilityReason] = []
    warning_reasons: list[UsabilityReason] = []
    usable = bool(guidance_usable)
    confidence = _confidence_from_support(heuristic_count, max_confidence)

    if not telemetry_summary:
        blocking_reasons.append(
            _reason(
                UsabilityReasonKind.BLOCKING,
                UsabilityBlockingReasonCode.COVERAGE_INSUFFICIENT,
                summary="Telemetry summary is missing, so scoring coverage is incomplete.",
                source_id="telemetry_summary",
                related_heuristics=heuristic_ids,
                confidence=0.9,
            )
        )
        usable = False
    if heuristic_count == 0:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.PARTIAL_COVERAGE,
                summary="Scoring evidence is present, but no runtime heuristics were derived.",
                source_id="heuristic_summary",
                related_heuristics=[],
                confidence=0.7,
            )
        )
    if max_confidence < 0.65:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.CONFIDENCE_FRAGMENTATION,
                summary="Scoring support is fragmented across low-confidence heuristics.",
                source_id="heuristic_summary",
                related_heuristics=heuristic_ids,
                confidence=max(0.5, max_confidence or 0.5),
            )
        )

    return UsabilityScopeAssessment(
        scope=UsabilityScope.SCORING,
        usable=usable,
        confidence=confidence,
        uncertainty_notes=[
            "Scoring usability is derived from canonical telemetry summaries.",
        ],
        provenance=[_provenance("runtime_usability", note="scoring_scope")],
        blocking_reasons=blocking_reasons,
        warning_reasons=warning_reasons,
    )


def _final_benchmark_scope(
    *,
    scoring_usable: bool,
    heuristic_ids: list[str],
    heuristic_count: int,
    max_confidence: float,
) -> UsabilityScopeAssessment:
    blocking_reasons: list[UsabilityReason] = []
    warning_reasons: list[UsabilityReason] = []
    usable = bool(scoring_usable and heuristic_count > 0 and max_confidence >= 0.45)
    confidence = _confidence_from_support(heuristic_count, max_confidence)

    if not scoring_usable:
        blocking_reasons.append(
            _reason(
                UsabilityReasonKind.BLOCKING,
                UsabilityBlockingReasonCode.BENCHMARK_CONTRACT_MISMATCH,
                summary="Final benchmark use is blocked when scoring usability is not satisfied.",
                source_id="scoring_scope",
                related_heuristics=heuristic_ids,
                confidence=max(0.7, max_confidence or 0.7),
            )
        )
    elif heuristic_count == 0:
        blocking_reasons.append(
            _reason(
                UsabilityReasonKind.BLOCKING,
                UsabilityBlockingReasonCode.COVERAGE_INSUFFICIENT,
                summary="Final benchmark use requires at least some heuristic support.",
                source_id="heuristic_summary",
                related_heuristics=[],
                confidence=0.8,
            )
        )

    if heuristic_count == 0:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.REVIEW_RECOMMENDED,
                summary="Final benchmark usability is not yet supported by runtime heuristics.",
                source_id="heuristic_summary",
                related_heuristics=[],
                confidence=0.7,
            )
        )
    elif heuristic_count == 1:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.NARROW_SUPPORT,
                summary="Final benchmark usability rests on a narrow heuristic signature.",
                source_id="heuristic_summary",
                related_heuristics=heuristic_ids,
                confidence=0.65,
            )
        )
    if max_confidence < 0.75:
        warning_reasons.append(
            _reason(
                UsabilityReasonKind.WARNING,
                UsabilityWarningReasonCode.REVIEW_RECOMMENDED,
                summary="Final benchmark usability still benefits from review.",
                source_id="heuristic_summary",
                related_heuristics=heuristic_ids,
                confidence=max(0.5, max_confidence or 0.5),
            )
        )

    return UsabilityScopeAssessment(
        scope=UsabilityScope.FINAL_BENCHMARK,
        usable=usable,
        confidence=confidence,
        uncertainty_notes=[
            "Final benchmark usability is emitted from runtime evidence only.",
        ],
        provenance=[_provenance("runtime_usability", note="final_benchmark_scope")],
        blocking_reasons=blocking_reasons,
        warning_reasons=warning_reasons,
    )


def build_runtime_usability_assessment(
    runtime_evidence: Mapping[str, Any] | None,
) -> UsabilityAssessment:
    """Build a first-class usability assessment from canonical runtime evidence."""
    payload = runtime_evidence if isinstance(runtime_evidence, Mapping) else {}
    runtime_context = payload.get("runtime_context", {})
    telemetry_summary = payload.get("telemetry_summary", {})
    heuristic_summary = payload.get("heuristic_summary", {})
    heuristic_ids = _heuristic_ids(payload)
    heuristic_count = len(heuristic_ids)
    max_confidence = 0.0
    if isinstance(heuristic_summary, Mapping):
        try:
            max_confidence = float(heuristic_summary.get("max_confidence", 0.0) or 0.0)
        except (TypeError, ValueError):
            max_confidence = 0.0

    guidance = _guidance_scope(
        runtime_context=runtime_context if isinstance(runtime_context, Mapping) else {},
        telemetry_summary=telemetry_summary if isinstance(telemetry_summary, Mapping) else {},
        heuristic_ids=heuristic_ids,
        heuristic_count=heuristic_count,
        max_confidence=max_confidence,
    )
    scoring = _scoring_scope(
        guidance_usable=guidance.usable,
        telemetry_summary=telemetry_summary if isinstance(telemetry_summary, Mapping) else {},
        heuristic_ids=heuristic_ids,
        heuristic_count=heuristic_count,
        max_confidence=max_confidence,
    )
    final_benchmark = _final_benchmark_scope(
        scoring_usable=scoring.usable,
        heuristic_ids=heuristic_ids,
        heuristic_count=heuristic_count,
        max_confidence=max_confidence,
    )

    return UsabilityAssessment(
        assessment_id=_ASSESSMENT_ID,
        family="",
        task_intent=_TASK_INTENT,
        heuristic_signature=heuristic_ids,
        required_contracts_checked=list(_REQUIRED_CONTRACTS),
        usable_for_guidance=guidance.usable,
        usable_for_scoring=scoring.usable,
        usable_for_final_benchmark=final_benchmark.usable,
        confidence=max(guidance.confidence, scoring.confidence, final_benchmark.confidence),
        uncertainty_notes=[
            "Runtime usability is derived from canonical runtime evidence only.",
        ],
        provenance=[_provenance("runtime_usability", note="assessment")],
        guidance=guidance,
        scoring=scoring,
        final_benchmark=final_benchmark,
    )


def serialize_runtime_usability_assessment(
    runtime_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return a JSON-compatible runtime usability artifact."""
    return build_runtime_usability_assessment(runtime_evidence).model_dump(mode="json")

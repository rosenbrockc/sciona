"""Deterministic admissibility contracts and cheap rule evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from sciona.architect.handoff import CDGExport
from sciona.architect.semantic_graph import (
    SemanticBoundaryKind,
    SemanticCDG,
    SemanticDataKind,
    SemanticLossClass,
    project_semantic_cdg,
)
from sciona.principal.runtime_context import (
    resolve_canonical_runtime_context,
    summarize_events,
    summarize_runtime_context,
    summarize_waveform,
)


class AdmissibilityDisposition(str, Enum):
    """Outcome classes for admissibility decisions."""

    PASS = "pass"
    HARD_REJECT = "hard_reject"
    SOFT_WARN = "soft_warn"
    ROUTE_TO_REFINEMENT = "route_to_refinement"


@dataclass(frozen=True)
class AdmissibilityDecision:
    """A single deterministic admissibility decision."""

    rule_id: str
    disposition: AdmissibilityDisposition
    summary: str
    severity: float
    evidence: str = ""
    metric_name: str = ""
    observed_value: float | None = None
    threshold: float | None = None
    family: str = ""
    suggested_refinement: str = ""


@dataclass
class AdmissibilityContext:
    """Cheap runtime context available to admissibility rules."""

    planning_artifact: dict[str, Any] | None = None
    semantic_cdg: SemanticCDG | None = None
    telemetry: dict[str, Any] = field(default_factory=dict)
    runtime_context: dict[str, Any] = field(default_factory=dict)
    family: str = ""

    def metric(self, path: str) -> Any:
        """Fetch a dotted metric path from telemetry or runtime context."""
        for source in (self.telemetry, self.runtime_context):
            value: Any = source
            found = True
            for part in path.split("."):
                if not isinstance(value, dict) or part not in value:
                    found = False
                    break
                value = value[part]
            if found:
                return value
        return None


@dataclass(frozen=True)
class AdmissibilityReport:
    """Aggregate deterministic admissibility result for one candidate."""

    decisions: tuple[AdmissibilityDecision, ...]

    @property
    def hard_rejected(self) -> bool:
        return any(
            decision.disposition == AdmissibilityDisposition.HARD_REJECT
            for decision in self.decisions
        )

    @property
    def routed_to_refinement(self) -> bool:
        return any(
            decision.disposition == AdmissibilityDisposition.ROUTE_TO_REFINEMENT
            for decision in self.decisions
        )

    @property
    def warnings(self) -> tuple[AdmissibilityDecision, ...]:
        return tuple(
            decision
            for decision in self.decisions
            if decision.disposition == AdmissibilityDisposition.SOFT_WARN
        )

    def decision_payloads(self) -> list[dict[str, Any]]:
        """Serialize per-rule decisions for runtime artifacts and trial history."""
        return [
            {
                "rule_id": decision.rule_id,
                "disposition": decision.disposition.value,
                "summary": decision.summary,
                "severity": decision.severity,
                "evidence": decision.evidence,
                "metric_name": decision.metric_name,
                "observed_value": decision.observed_value,
                "threshold": decision.threshold,
                "family": decision.family,
                "suggested_refinement": decision.suggested_refinement,
            }
            for decision in self.decisions
        ]

    def summary(self) -> dict[str, Any]:
        """Serialize a compact structured summary for later trial-history wiring."""
        return {
            "hard_rejected": self.hard_rejected,
            "routed_to_refinement": self.routed_to_refinement,
            "decision_count": len(self.decisions),
            "decisions": self.decision_payloads(),
            "hard_reject_rule_ids": [
                decision.rule_id
                for decision in self.decisions
                if decision.disposition == AdmissibilityDisposition.HARD_REJECT
            ],
            "warning_rule_ids": [
                decision.rule_id
                for decision in self.decisions
                if decision.disposition == AdmissibilityDisposition.SOFT_WARN
            ],
            "refinement_rule_ids": [
                decision.rule_id
                for decision in self.decisions
                if decision.disposition
                == AdmissibilityDisposition.ROUTE_TO_REFINEMENT
            ],
        }


@runtime_checkable
class AdmissibilityRule(Protocol):
    """Protocol for deterministic admissibility rules."""

    rule_id: str

    def evaluate(self, context: AdmissibilityContext) -> list[AdmissibilityDecision]:
        """Return zero or more decisions for the current candidate."""
        ...


class RequiredRuntimeKeysRule:
    """Hard-reject when required runtime context is absent."""

    def __init__(self, required_keys: list[str], *, rule_id: str = "required_runtime_keys"):
        self.rule_id = rule_id
        self._required_keys = tuple(required_keys)

    def evaluate(self, context: AdmissibilityContext) -> list[AdmissibilityDecision]:
        missing = [
            key for key in self._required_keys if context.metric(key) is None
        ]
        if not missing:
            return []
        return [
            AdmissibilityDecision(
                rule_id=self.rule_id,
                disposition=AdmissibilityDisposition.HARD_REJECT,
                summary="Candidate is missing required runtime context.",
                severity=1.0,
                evidence=f"Missing runtime keys: {', '.join(missing)}",
                family=context.family,
            )
        ]


class MinimumCountPerDurationRule:
    """Hard-reject catastrophic sparsity relative to observed duration."""

    def __init__(
        self,
        *,
        count_metric: str,
        duration_metric: str,
        min_per_minute: float,
        rule_id: str = "minimum_count_per_duration",
    ) -> None:
        self.rule_id = rule_id
        self._count_metric = count_metric
        self._duration_metric = duration_metric
        self._min_per_minute = float(min_per_minute)

    def evaluate(self, context: AdmissibilityContext) -> list[AdmissibilityDecision]:
        count = context.metric(self._count_metric)
        duration_seconds = context.metric(self._duration_metric)
        if count is None or duration_seconds is None:
            return []
        duration_minutes = float(duration_seconds) / 60.0
        if duration_minutes <= 0:
            return []
        observed = float(count) / duration_minutes
        if observed >= self._min_per_minute:
            return []
        return [
            AdmissibilityDecision(
                rule_id=self.rule_id,
                disposition=AdmissibilityDisposition.HARD_REJECT,
                summary="Candidate output is catastrophically sparse for its duration.",
                severity=1.0,
                evidence=(
                    f"Observed {float(count):.1f} items over {duration_minutes:.1f} min "
                    f"({observed:.2f}/min)."
                ),
                metric_name=self._count_metric,
                observed_value=observed,
                threshold=self._min_per_minute,
                family=context.family,
            )
        ]


class RootBoundaryLossRule:
    """Hard-reject disallowed lossy transitions directly off a root boundary."""

    def __init__(
        self,
        *,
        protected_data_kind: SemanticDataKind = SemanticDataKind.WAVEFORM,
        rule_id: str = "root_boundary_loss",
    ) -> None:
        self.rule_id = rule_id
        self._protected_data_kind = protected_data_kind

    def evaluate(self, context: AdmissibilityContext) -> list[AdmissibilityDecision]:
        semantic = context.semantic_cdg
        if semantic is None:
            return []

        protected_boundaries = {
            boundary.boundary_id
            for boundary in semantic.boundaries
            if boundary.kind == SemanticBoundaryKind.ROOT_INPUT
        }
        decisions: list[AdmissibilityDecision] = []
        for edge in semantic.edges:
            if edge.source_id not in protected_boundaries:
                continue
            if edge.data_kind != self._protected_data_kind:
                continue
            if edge.loss_class != SemanticLossClass.LOSSY:
                continue
            decisions.append(
                AdmissibilityDecision(
                    rule_id=self.rule_id,
                    disposition=AdmissibilityDisposition.HARD_REJECT,
                    summary="Candidate destroys protected boundary information too early.",
                    severity=1.0,
                    evidence=(
                        f"Root boundary edge {edge.source_id} -> {edge.target_id} "
                        f"is marked {edge.loss_class.value} for {edge.data_kind.value}."
                    ),
                    family=context.family,
                )
            )
        return decisions


class ThresholdMetricRule:
    """Generic cheap metric gate for warnings or refinement routing."""

    def __init__(
        self,
        *,
        rule_id: str,
        metric_name: str,
        threshold: float,
        disposition: AdmissibilityDisposition,
        summary: str,
        suggested_refinement: str = "",
    ) -> None:
        self.rule_id = rule_id
        self._metric_name = metric_name
        self._threshold = float(threshold)
        self._disposition = disposition
        self._summary = summary
        self._suggested_refinement = suggested_refinement

    def evaluate(self, context: AdmissibilityContext) -> list[AdmissibilityDecision]:
        observed = context.metric(self._metric_name)
        if observed is None:
            return []
        value = float(observed)
        if value <= self._threshold:
            return []
        severity = min(1.0, value / max(self._threshold, 1e-9))
        return [
            AdmissibilityDecision(
                rule_id=self.rule_id,
                disposition=self._disposition,
                summary=self._summary,
                severity=severity,
                evidence=(
                    f"Observed {self._metric_name}={value:.3f}, threshold "
                    f"{self._threshold:.3f}."
                ),
                metric_name=self._metric_name,
                observed_value=value,
                threshold=self._threshold,
                family=context.family,
                suggested_refinement=self._suggested_refinement,
            )
        ]


class AdmissibilityEvaluator:
    """Evaluate a deterministic admissibility rule bundle."""

    def __init__(self, rules: list[AdmissibilityRule] | None = None) -> None:
        self._rules = list(rules or [])

    def evaluate(self, context: AdmissibilityContext) -> AdmissibilityReport:
        decisions: list[AdmissibilityDecision] = []
        for rule in self._rules:
            decisions.extend(rule.evaluate(context))
        decisions.sort(key=lambda decision: (-decision.severity, decision.rule_id))
        return AdmissibilityReport(decisions=tuple(decisions))


def infer_admissibility_family(
    planning_artifact: dict[str, Any] | None,
    *,
    fallback: str = "",
) -> str:
    """Resolve a family hint from planning metadata when available."""
    artifact = planning_artifact or {}
    if hasattr(artifact, "model_dump"):
        artifact = artifact.model_dump(mode="json")
    if not isinstance(artifact, dict):
        return fallback
    for key in ("family_hint", "paradigm"):
        value = str(artifact.get(key, "")).strip()
        if value:
            return value
    return fallback


def build_admissibility_context(
    *,
    cdg: CDGExport,
    planning_artifact: dict[str, Any] | None,
    runtime_artifacts: dict[str, Any] | None,
    family: str = "",
) -> AdmissibilityContext:
    """Build a cheap, deterministic admissibility context from runtime artifacts."""
    artifacts = runtime_artifacts if isinstance(runtime_artifacts, dict) else {}
    signal_data = (
        dict(artifacts.get("signal_data", {}))
        if isinstance(artifacts.get("signal_data", {}), dict)
        else {}
    )
    intermediates = (
        dict(artifacts.get("intermediates", {}))
        if isinstance(artifacts.get("intermediates", {}), dict)
        else {}
    )

    stored_runtime_context = artifacts.get("runtime_context", {})
    runtime_context = (
        dict(stored_runtime_context)
        if isinstance(stored_runtime_context, dict)
        else {}
    )
    stored_summary = artifacts.get("telemetry_summary", {})
    telemetry = dict(stored_summary) if isinstance(stored_summary, dict) else {}
    canonical = resolve_canonical_runtime_context(signal_data)
    if not runtime_context:
        runtime_context = summarize_runtime_context(canonical)

    sampling_rate: float | None = None
    if isinstance(runtime_context.get("sampling_rate"), (int, float)):
        sampling_rate = float(runtime_context["sampling_rate"])
    else:
        sampling_ref = canonical.canonical_inputs.get("sampling_rate")
        if sampling_ref is not None:
            raw_value = signal_data.get(sampling_ref.raw_key)
            try:
                sampling_rate = float(raw_value)
                runtime_context["sampling_rate"] = sampling_rate
            except Exception:
                sampling_rate = None

    signal_ref = canonical.canonical_inputs.get("signal")
    if signal_ref is not None and "signal" not in telemetry:
        signal_values = signal_data.get(signal_ref.raw_key)
        if signal_values is not None:
            telemetry["signal"] = summarize_waveform(signal_values)
            runtime_context["signal"] = signal_ref.raw_key
    elif signal_ref is not None:
        runtime_context.setdefault("signal", signal_ref.raw_key)

    events = intermediates.get("events")
    if events is not None and "events" not in telemetry:
        duration_seconds: float | None = None
        if signal_ref is not None and sampling_rate and sampling_rate > 0:
            signal_values = signal_data.get(signal_ref.raw_key)
            try:
                duration_seconds = float(len(signal_values)) / float(sampling_rate)
            except Exception:
                duration_seconds = None
        telemetry["events"] = summarize_events(
            events,
            sampling_rate=sampling_rate,
            duration_seconds=duration_seconds,
        )
        if duration_seconds is not None:
            telemetry["events"]["duration_seconds"] = duration_seconds
    elif "events" in telemetry:
        duration_seconds = telemetry["events"].get("duration_seconds")
        if duration_seconds is not None and not isinstance(duration_seconds, (int, float)):
            telemetry["events"].pop("duration_seconds", None)

    return AdmissibilityContext(
        planning_artifact=planning_artifact,
        semantic_cdg=project_semantic_cdg(cdg),
        telemetry=telemetry,
        runtime_context=runtime_context,
        family=family or infer_admissibility_family(planning_artifact),
    )


def default_admissibility_rules(*, family: str = "") -> list[AdmissibilityRule]:
    """Return the default deterministic admissibility bundle for a family."""
    rules: list[AdmissibilityRule] = [RootBoundaryLossRule()]

    family_key = family.strip().lower()
    if family_key in {"signal_event_rate", "signal_detect_measure"}:
        rules.extend(
            [
                RequiredRuntimeKeysRule(
                    ["sampling_rate"],
                    rule_id="needs_sampling_rate",
                ),
                MinimumCountPerDurationRule(
                    count_metric="events.count",
                    duration_metric="events.duration_seconds",
                    min_per_minute=20.0,
                    rule_id="minimum_event_density",
                ),
                ThresholdMetricRule(
                    rule_id="unstable_event_intervals",
                    metric_name="events.outlier_fraction",
                    threshold=0.15,
                    disposition=AdmissibilityDisposition.ROUTE_TO_REFINEMENT,
                    summary=(
                        "Detected event intervals are unstable enough to require refinement."
                    ),
                    suggested_refinement="insert_outlier_rejection_after_detection",
                ),
            ]
        )
    return rules


def default_admissibility_evaluator(*, family: str = "") -> AdmissibilityEvaluator:
    """Construct the default admissibility evaluator for a family."""
    return AdmissibilityEvaluator(default_admissibility_rules(family=family))


def default_structural_admissibility_evaluator() -> AdmissibilityEvaluator:
    """Return the cheap admissibility bundle safe before synthesis/evaluation."""
    return AdmissibilityEvaluator([RootBoundaryLossRule()])

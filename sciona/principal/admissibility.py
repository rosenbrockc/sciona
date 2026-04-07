"""Deterministic admissibility contracts and cheap rule evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from sciona.architect.semantic_graph import (
    SemanticBoundaryKind,
    SemanticCDG,
    SemanticDataKind,
    SemanticLossClass,
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

    def summary(self) -> dict[str, Any]:
        """Serialize a compact structured summary for later trial-history wiring."""
        return {
            "hard_rejected": self.hard_rejected,
            "routed_to_refinement": self.routed_to_refinement,
            "decision_count": len(self.decisions),
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

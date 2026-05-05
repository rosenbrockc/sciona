"""Diagnostic-driven CDG topology expansion via DPO graph rewriting.

The expansion system is domain-agnostic by design:

  - **ExpansionRuleSet** is a protocol that any domain can implement.
  - **ExpansionEngine** collects diagnostics from ALL registered rule sets,
    regardless of domain, sorts by severity, and applies triggered rules.
  - The DPO pattern matching in :class:`GraphRewriter` is the final guard —
    if a rule's LHS doesn't match the CDG topology it fails gracefully.

This enables cross-domain expansion: a signal-processing SQI rule can fire
on a statistical-inference CDG if that CDG happens to contain signal nodes,
and vice versa.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from sciona.architect.graph_rewriter import GraphRewriter, RewriteRule
from sciona.architect.handoff import CDGExport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data contracts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExpansionDiagnostic:
    """A diagnostic signal recommending a specific expansion rule.

    Diagnostics are *pure* functions of the CDG and runtime intermediates.
    They must be deterministic and side-effect free so that expansion
    outcomes are reproducible.
    """

    rule_name: str  # which RewriteRule to apply
    severity: float  # 0.0 = no action … 1.0 = critical
    evidence: str  # human-readable justification
    metric_name: str  # which metric triggered this
    metric_value: float  # measured value
    threshold: float  # the threshold that was exceeded
    source_domain: str = ""  # originating rule-set domain
    asset_id: str = ""
    asset_version: str = ""
    asset_family: str = ""
    asset_source_kind: str = ""
    asset_review_status: str = ""
    asset_operation: str = ""
    asset_operation_id: str = ""
    asset_operation_type: str = ""
    asset_applies_to: str = ""
    asset_migration_readiness_status: str = ""
    asset_migration_readiness_ready: bool = False
    asset_migration_readiness_check_count: int = 0
    asset_migration_readiness_required_check_count: int = 0


@dataclass
class ExpansionContext:
    """Runtime context available to diagnostics.

    Passed by the Principal after pipeline execution so that diagnostics
    can inspect intermediate outputs and evaluation results.
    """

    intermediates: dict[str, Any] = field(default_factory=dict)
    eval_result: dict[str, Any] | None = None
    runtime_inputs: dict[str, Any] | None = None
    signal_data: dict[str, Any] | None = None
    runtime_evidence: dict[str, Any] | None = None
    planning_artifact: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        """Keep the deprecated signal_data alias aligned with runtime_inputs."""
        if self.runtime_inputs is None and self.signal_data is not None:
            self.runtime_inputs = dict(self.signal_data)
        elif self.signal_data is None and self.runtime_inputs is not None:
            self.signal_data = dict(self.runtime_inputs)


@dataclass(frozen=True)
class ExpansionResult:
    """Outcome of an expansion attempt."""

    cdg: CDGExport
    applied_rules: tuple[str, ...]
    diagnostics: tuple[ExpansionDiagnostic, ...]
    expanded: bool
    applied_assets: tuple[dict[str, Any], ...] = ()


# ---------------------------------------------------------------------------
# Rule-set protocol (domain-agnostic)
# ---------------------------------------------------------------------------


@runtime_checkable
class ExpansionRuleSet(Protocol):
    """Domain-specific bundle of diagnostics and DPO expansion rules.

    There is intentionally **no** ``matches()`` gate.  Diagnostics decide
    relevance: if a rule set's diagnostics return nothing for a given CDG,
    no rules from that set are tried.  This allows cross-domain expansion
    without requiring every rule set to declare which graph shapes it owns.
    """

    @property
    def name(self) -> str: ...

    @property
    def domain(self) -> str: ...

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        """Return diagnostics for *cdg*.  Empty list → nothing to expand."""
        ...

    def rules(self) -> list[RewriteRule]:
        """Return the full catalog of DPO rules this set can provide."""
        ...


# ---------------------------------------------------------------------------
# Expansion engine
# ---------------------------------------------------------------------------


class ExpansionEngine:
    """Orchestrates diagnostic-driven CDG topology expansion.

    Lifecycle (called from the Principal trial loop)::

        engine = ExpansionEngine(rule_sets, rewriter)
        result = engine.expand(cdg, context)
        if result.expanded:
            cdg = result.cdg   # use expanded graph

    The engine is stateless between calls — all context is passed in.
    """

    def __init__(
        self,
        rule_sets: list[ExpansionRuleSet] | None = None,
        rewriter: GraphRewriter | None = None,
        *,
        activation_threshold: float = 0.0,
    ):
        self._rule_sets: list[ExpansionRuleSet] = list(rule_sets or [])
        self._rewriter = rewriter or GraphRewriter()
        self._activation_threshold = activation_threshold
        self._rebuild_index()

    # -- public API --------------------------------------------------------

    def register(self, rule_set: ExpansionRuleSet) -> None:
        """Add a rule set (from any domain) at runtime."""
        self._rule_sets.append(rule_set)
        self._rebuild_index()

    def expand(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> ExpansionResult:
        """Run all diagnostics, apply triggered rules, return result.

        Rules are applied in severity order (highest first).  Each rule is
        applied at most once.  If a rule's LHS pattern does not match the
        (possibly already-expanded) CDG, it is silently skipped.
        """
        all_diagnostics: list[ExpansionDiagnostic] = []
        applied_rules: list[str] = []
        applied_assets: list[dict[str, Any]] = []

        # Collect diagnostics from ALL rule sets (cross-domain).
        for rs in self._rule_sets:
            try:
                diags = rs.diagnose(cdg, context)
                all_diagnostics.extend(diags)
            except Exception:
                logger.warning(
                    "Diagnostics failed for rule set '%s'",
                    rs.name,
                    exc_info=True,
                )

        # Sort by severity — most critical expansions first.
        active = sorted(
            (d for d in all_diagnostics if d.severity >= self._activation_threshold),
            key=lambda d: -d.severity,
        )

        applied_set: set[str] = set()
        for diag in active:
            if diag.rule_name in applied_set:
                continue

            rule = self._rule_index.get(diag.rule_name)
            if rule is None:
                logger.warning(
                    "Diagnostic references unknown rule '%s'", diag.rule_name
                )
                continue

            result = self._rewriter.apply_rule(rule, cdg)
            if result.is_failure:
                logger.debug(
                    "Rule '%s' did not apply: %s", diag.rule_name, result.error
                )
                continue

            cdg = result.unwrap()
            applied_rules.append(diag.rule_name)
            applied_set.add(diag.rule_name)
            if diag.asset_id:
                asset_summary = {
                    "asset_id": diag.asset_id,
                    "asset_version": diag.asset_version,
                    "asset_family": diag.asset_family,
                    "asset_source_kind": diag.asset_source_kind,
                    "asset_review_status": diag.asset_review_status,
                    "asset_operation": diag.asset_operation,
                    "asset_operation_id": diag.asset_operation_id,
                    "asset_operation_type": diag.asset_operation_type,
                    "asset_applies_to": diag.asset_applies_to,
                    "asset_migration_readiness_status": (
                        diag.asset_migration_readiness_status
                    ),
                    "asset_migration_readiness_ready": (
                        diag.asset_migration_readiness_ready
                    ),
                    "asset_migration_readiness_check_count": (
                        diag.asset_migration_readiness_check_count
                    ),
                    "asset_migration_readiness_required_check_count": (
                        diag.asset_migration_readiness_required_check_count
                    ),
                    "rule_name": diag.rule_name,
                }
                if asset_summary not in applied_assets:
                    applied_assets.append(asset_summary)
            logger.info(
                "Expansion '%s' applied (severity=%.2f, %s=%.3f > %.3f)",
                diag.rule_name,
                diag.severity,
                diag.metric_name,
                diag.metric_value,
                diag.threshold,
            )

        return ExpansionResult(
            cdg=cdg,
            applied_rules=tuple(applied_rules),
            applied_assets=tuple(applied_assets),
            diagnostics=tuple(all_diagnostics),
            expanded=len(applied_rules) > 0,
        )

    # -- internals ---------------------------------------------------------

    def _rebuild_index(self) -> None:
        self._rule_index: dict[str, RewriteRule] = {}
        for rs in self._rule_sets:
            for rule in rs.rules():
                if rule.name in self._rule_index:
                    logger.debug(
                        "Rule '%s' overridden by rule set '%s'",
                        rule.name,
                        rs.name,
                    )
                self._rule_index[rule.name] = rule

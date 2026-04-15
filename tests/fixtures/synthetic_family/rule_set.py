"""Synthetic-family ExpansionRuleSet for matcher engine-level tests."""

from __future__ import annotations

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

SYNTH_DOMAIN = "synthetic_family"


def _atomic(
    node_id: str,
    name: str,
    *,
    matched_primitive: str | None = None,
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=[IOSpec(name="in", type_desc="synth_payload")],
        outputs=[IOSpec(name="out", type_desc="synth_payload")],
        type_signature=f"{name} -> synth_payload",
    )


def _edge(source_id: str, target_id: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name="out",
        input_name="in",
        source_type="synth_payload",
        target_type="synth_payload",
    )


# ---------------------------------------------------------------------------
# Rule builders
# ---------------------------------------------------------------------------


def build_insert_quality_check_rule() -> RewriteRule:
    """Interpose ``measure_synth_quality`` on any source→process edge."""
    source = _atomic("source", "synth_source", matched_primitive="synth_source")
    process = _atomic("process", "synth_process", matched_primitive="synth_process")
    lhs = CDGExport(nodes=[source, process], edges=[_edge("source", "process")])
    interface = CDGExport(nodes=[source, process], edges=[])

    checker = _atomic(
        "quality_check",
        "Measure Synth Quality",
        matched_primitive="measure_synth_quality",
    )
    rhs = CDGExport(
        nodes=[source, checker, process],
        edges=[_edge("source", "quality_check"), _edge("quality_check", "process")],
    )

    return RewriteRule(
        name="insert_quality_check_after_source",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={"source": "source", "process": "process"}, edge_map={}
        ),
        r_morphism=Morphism(
            node_map={"source": "source", "process": "process"}, edge_map={}
        ),
        priority=2,
    )


def build_wrap_sink_with_audit_rule() -> RewriteRule:
    """Interpose ``audit_sink`` on any process→sink edge."""
    process = _atomic("process", "synth_process", matched_primitive="synth_process")
    sink = _atomic("sink", "synth_sink", matched_primitive="synth_sink")
    lhs = CDGExport(nodes=[process, sink], edges=[_edge("process", "sink")])
    interface = CDGExport(nodes=[process, sink], edges=[])

    audit = _atomic("audit", "Audit Sink", matched_primitive="audit_sink")
    rhs = CDGExport(
        nodes=[process, audit, sink],
        edges=[_edge("process", "audit"), _edge("audit", "sink")],
    )

    return RewriteRule(
        name="wrap_sink_with_audit",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={"process": "process", "sink": "sink"}, edge_map={}
        ),
        r_morphism=Morphism(
            node_map={"process": "process", "sink": "sink"}, edge_map={}
        ),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


_QUALITY_THRESHOLD = 0.5
_ERROR_RATE_THRESHOLD = 0.1


def _diagnose_quality(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    score = (context.intermediates or {}).get("synth_quality_score")
    if score is None:
        return None
    try:
        value = float(score)
    except (TypeError, ValueError):
        return None
    if value >= _QUALITY_THRESHOLD:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_quality_check_after_source",
        severity=min(1.0, (_QUALITY_THRESHOLD - value) / _QUALITY_THRESHOLD),
        evidence=(
            f"synth_quality_score={value:.3f} below {_QUALITY_THRESHOLD} "
            f"threshold — interpose quality measurement"
        ),
        metric_name="synth_quality_score",
        metric_value=value,
        threshold=_QUALITY_THRESHOLD,
        source_domain=SYNTH_DOMAIN,
    )


def _diagnose_error_rate(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    rate = (context.intermediates or {}).get("synth_error_rate")
    if rate is None:
        return None
    try:
        value = float(rate)
    except (TypeError, ValueError):
        return None
    if value <= _ERROR_RATE_THRESHOLD:
        return None
    return ExpansionDiagnostic(
        rule_name="wrap_sink_with_audit",
        severity=min(1.0, (value - _ERROR_RATE_THRESHOLD) / _ERROR_RATE_THRESHOLD),
        evidence=(
            f"synth_error_rate={value:.3f} exceeds {_ERROR_RATE_THRESHOLD} "
            f"threshold — wrap sink with audit"
        ),
        metric_name="synth_error_rate",
        metric_value=value,
        threshold=_ERROR_RATE_THRESHOLD,
        source_domain=SYNTH_DOMAIN,
    )


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SyntheticFamilyExpansionRuleSet:
    """Minimal ExpansionRuleSet used exclusively by matcher engine tests.

    Provides two insert-on-edge DPO rules and two corresponding
    diagnostics. The semantics are deliberately trivial so tests can
    assert engine behaviour (ordering, dedup, aggregation, asset
    loading) without coupling to any real family's numerical content.
    """

    name = "synthetic_family"
    domain = SYNTH_DOMAIN

    def __init__(self) -> None:
        self._rules = [
            build_insert_quality_check_rule(),
            build_wrap_sink_with_audit_rule(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        quality = _diagnose_quality(cdg, context)
        if quality is not None:
            diagnostics.append(quality)
        error_rate = _diagnose_error_rate(cdg, context)
        if error_rate is not None:
            diagnostics.append(error_rate)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

"""Expansion rules for the Signal Detect Measure family.

Signal Detect Measure skeleton topology (3 nodes, linear pipeline):

    Filter Signal For Detection → Detect Peaks In Signal → Compute Event Rate

Expansion insertion points:
  - Before Filter Signal For Detection: SNR estimation
  - After Filter Signal For Detection: peak threshold sensitivity
  - After Detect Peaks In Signal: event rate stationarity check
  - After Compute Event Rate: false positive rate estimation
"""

from __future__ import annotations

import logging

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)

logger = logging.getLogger(__name__)

_DOMAIN = "signal_detect_measure"

_FILTER_SIGNAL = "Filter Signal For Detection"
_DETECT_PEAKS = "Detect Peaks In Signal"
_COMPUTE_RATE = "Compute Event Rate"


def _node(
    node_id: str, name: str, concept_type: ConceptType, *,
    matched_primitive: str | None = None, inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None, description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id, name=name, description=description or name,
        concept_type=concept_type, status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive, inputs=inputs or [],
        outputs=outputs or [], type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str, target_id: str, output_name: str = "out",
    input_name: str = "in", type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id, target_id=target_id, output_name=output_name,
        input_name=input_name, source_type=type_desc, target_type=type_desc,
    )


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_snr_estimation() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    filter_sig = _node("filter_sig", _FILTER_SIGNAL, ConceptType.SIGNAL_FILTER)
    lhs = CDGExport(nodes=[src, filter_sig], edges=[_edge("src", "filter_sig")])
    interface = CDGExport(nodes=[src, filter_sig], edges=[])

    snr = _node(
        "snr", "Estimate SNR", ConceptType.ANALYSIS,
        matched_primitive="estimate_snr",
        inputs=[IOSpec(name="signal", type_desc="ndarray"), IOSpec(name="noise_floor", type_desc="float")],
        outputs=[IOSpec(name="snr_db", type_desc="float"), IOSpec(name="is_sufficient", type_desc="bool")],
        description="Estimate signal-to-noise ratio.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[src, snr, filter_sig], edges=[_edge("src", "snr"), _edge("snr", "filter_sig")])

    return RewriteRule(
        name="insert_snr_estimation_before_filter", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "filter_sig": "filter_sig"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "filter_sig": "filter_sig"}, edge_map={}),
        priority=3,
    )


def _build_insert_threshold_sensitivity() -> RewriteRule:
    filter_sig = _node("filter_sig", _FILTER_SIGNAL, ConceptType.SIGNAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[filter_sig, sink], edges=[_edge("filter_sig", "sink")])
    interface = CDGExport(nodes=[filter_sig, sink], edges=[])

    sensitivity = _node(
        "sensitivity", "Analyze Peak Threshold Sensitivity", ConceptType.ANALYSIS,
        matched_primitive="analyze_peak_threshold_sensitivity",
        inputs=[IOSpec(name="peaks", type_desc="ndarray"), IOSpec(name="threshold", type_desc="float")],
        outputs=[IOSpec(name="sensitivity", type_desc="float"), IOSpec(name="is_stable", type_desc="bool")],
        description="Analyze how sensitive detection count is to threshold changes.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[filter_sig, sensitivity, sink], edges=[_edge("filter_sig", "sensitivity"), _edge("sensitivity", "sink")])

    return RewriteRule(
        name="insert_threshold_sensitivity_after_filter", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"filter_sig": "filter_sig", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"filter_sig": "filter_sig", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_rate_stationarity() -> RewriteRule:
    detect = _node("detect", _DETECT_PEAKS, ConceptType.DATA_EXTRACTION)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[detect, sink], edges=[_edge("detect", "sink")])
    interface = CDGExport(nodes=[detect, sink], edges=[])

    stationarity = _node(
        "stationarity", "Check Event Rate Stationarity", ConceptType.ANALYSIS,
        matched_primitive="check_event_rate_stationarity",
        inputs=[IOSpec(name="event_times", type_desc="ndarray"), IOSpec(name="n_bins", type_desc="int")],
        outputs=[IOSpec(name="coefficient_of_variation", type_desc="float"), IOSpec(name="is_stationary", type_desc="bool")],
        description="Check whether the event rate is stationary over time.",
        type_signature="ndarray, int -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[detect, stationarity, sink], edges=[_edge("detect", "stationarity"), _edge("stationarity", "sink")])

    return RewriteRule(
        name="insert_rate_stationarity_after_detect", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"detect": "detect", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"detect": "detect", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_false_positive_estimation() -> RewriteRule:
    rate = _node("rate", _COMPUTE_RATE, ConceptType.ANALYSIS)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[rate, sink], edges=[_edge("rate", "sink")])
    interface = CDGExport(nodes=[rate, sink], edges=[])

    fpr = _node(
        "fpr", "Estimate False Positive Rate", ConceptType.ANALYSIS,
        matched_primitive="estimate_false_positive_rate",
        inputs=[IOSpec(name="detected_amplitudes", type_desc="ndarray"), IOSpec(name="noise_std", type_desc="float"),
                IOSpec(name="threshold", type_desc="float")],
        outputs=[IOSpec(name="estimated_fpr", type_desc="float"), IOSpec(name="is_reliable", type_desc="bool")],
        description="Estimate the false positive detection rate.",
        type_signature="ndarray, float, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[rate, fpr, sink], edges=[_edge("rate", "fpr"), _edge("fpr", "sink")])

    return RewriteRule(
        name="insert_false_positive_estimation_after_rate", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"rate": "rate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"rate": "rate", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_snr(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    snr = intermediates.get("snr_db")
    if snr is None:
        return None
    try:
        s = float(snr)
    except (ValueError, TypeError):
        return None
    if s < 10.0:
        return ExpansionDiagnostic(
            rule_name="insert_snr_estimation_before_filter",
            severity=min(1.0, (10.0 - s) / 10.0), evidence=f"SNR {s:.1f} dB below 10 dB — detection reliability degraded",
            metric_name="snr_db", metric_value=s, threshold=10.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_threshold_sensitivity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    sens = intermediates.get("threshold_sensitivity")
    if sens is None:
        return None
    try:
        s = float(sens)
    except (ValueError, TypeError):
        return None
    if s > 0.2:
        return ExpansionDiagnostic(
            rule_name="insert_threshold_sensitivity_after_filter",
            severity=min(1.0, s * 2), evidence=f"Threshold sensitivity {s:.2f} exceeds 0.2 — unstable detection count",
            metric_name="threshold_sensitivity", metric_value=s, threshold=0.2, source_domain=_DOMAIN,
        )
    return None


def _diagnose_rate_stationarity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    cv = intermediates.get("event_rate_cv")
    if cv is None:
        return None
    try:
        c = float(cv)
    except (ValueError, TypeError):
        return None
    if c > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_rate_stationarity_after_detect",
            severity=min(1.0, c), evidence=f"Event rate CV {c:.2f} exceeds 0.5 — non-stationary rate",
            metric_name="event_rate_cv", metric_value=c, threshold=0.5, source_domain=_DOMAIN,
        )
    return None


def _diagnose_false_positive(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    fpr = intermediates.get("false_positive_rate")
    if fpr is None:
        return None
    try:
        f = float(fpr)
    except (ValueError, TypeError):
        return None
    if f > 0.05:
        return ExpansionDiagnostic(
            rule_name="insert_false_positive_estimation_after_rate",
            severity=min(1.0, f * 10), evidence=f"False positive rate {f:.3f} exceeds 0.05 — unreliable detections",
            metric_name="false_positive_rate", metric_value=f, threshold=0.05, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SignalDetectMeasureExpansionRuleSet:
    """Expansion rules for signal detection and measurement pipelines."""

    name = "signal_detect_measure"
    domain = "signal_detect_measure"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_snr_estimation(),
            _build_insert_threshold_sensitivity(),
            _build_insert_rate_stationarity(),
            _build_insert_false_positive_estimation(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_snr, _diagnose_threshold_sensitivity, _diagnose_rate_stationarity, _diagnose_false_positive]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

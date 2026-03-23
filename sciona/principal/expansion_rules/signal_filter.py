"""Expansion rules for the Signal Filter family (Butterworth, Chebyshev, FIR, Notch).

Signal Filter skeleton topology (4 nodes, branching):

    Design Filter → Validate Stability → Apply Filter
                                       → Frequency Response

Expansion insertion points:
  - After Design Filter: pole-zero stability analysis
  - After Validate Stability: passband ripple measurement
  - After Apply Filter: transient response detection
  - After Frequency Response: group delay variation analysis
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

_DOMAIN = "signal_filter"

_DESIGN_FILTER = "Design Filter"
_VALIDATE_STABILITY = "Validate Stability"
_APPLY_FILTER = "Apply Filter"
_FREQUENCY_RESPONSE = "Frequency Response"


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


def _build_insert_pole_stability_analysis() -> RewriteRule:
    design = _node("design", _DESIGN_FILTER, ConceptType.SIGNAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[design, sink], edges=[_edge("design", "sink")])
    interface = CDGExport(nodes=[design, sink], edges=[])

    stability = _node(
        "stability", "Analyze Pole Stability", ConceptType.SIGNAL_FILTER,
        matched_primitive="analyze_pole_stability",
        inputs=[IOSpec(name="poles", type_desc="ndarray"), IOSpec(name="margin", type_desc="float")],
        outputs=[IOSpec(name="max_pole_magnitude", type_desc="float"), IOSpec(name="is_stable", type_desc="bool")],
        description="Analyze filter stability from pole locations.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[design, stability, sink], edges=[_edge("design", "stability"), _edge("stability", "sink")])

    return RewriteRule(
        name="insert_pole_stability_analysis_after_design", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"design": "design", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"design": "design", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_passband_ripple_measurement() -> RewriteRule:
    validate = _node("validate", _VALIDATE_STABILITY, ConceptType.SIGNAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[validate, sink], edges=[_edge("validate", "sink")])
    interface = CDGExport(nodes=[validate, sink], edges=[])

    ripple = _node(
        "ripple", "Measure Passband Ripple", ConceptType.SIGNAL_FILTER,
        matched_primitive="measure_passband_ripple",
        inputs=[IOSpec(name="freq_response_db", type_desc="ndarray"), IOSpec(name="passband_mask", type_desc="ndarray")],
        outputs=[IOSpec(name="ripple_db", type_desc="float"), IOSpec(name="is_acceptable", type_desc="bool")],
        description="Measure peak-to-peak ripple in the filter passband.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[validate, ripple, sink], edges=[_edge("validate", "ripple"), _edge("ripple", "sink")])

    return RewriteRule(
        name="insert_passband_ripple_measurement_after_validate", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"validate": "validate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"validate": "validate", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_group_delay_analysis() -> RewriteRule:
    freq_resp = _node("freq_resp", _FREQUENCY_RESPONSE, ConceptType.SIGNAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[freq_resp, sink], edges=[_edge("freq_resp", "sink")])
    interface = CDGExport(nodes=[freq_resp, sink], edges=[])

    delay = _node(
        "delay", "Analyze Group Delay Variation", ConceptType.SIGNAL_FILTER,
        matched_primitive="analyze_group_delay_variation",
        inputs=[IOSpec(name="group_delay", type_desc="ndarray")],
        outputs=[IOSpec(name="delay_variation", type_desc="float"), IOSpec(name="is_linear_phase", type_desc="bool")],
        description="Analyze group delay variation across frequency.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[freq_resp, delay, sink], edges=[_edge("freq_resp", "delay"), _edge("delay", "sink")])

    return RewriteRule(
        name="insert_group_delay_analysis_after_frequency_response", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"freq_resp": "freq_resp", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"freq_resp": "freq_resp", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _build_insert_transient_detection() -> RewriteRule:
    apply_f = _node("apply_f", _APPLY_FILTER, ConceptType.SIGNAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[apply_f, sink], edges=[_edge("apply_f", "sink")])
    interface = CDGExport(nodes=[apply_f, sink], edges=[])

    transient = _node(
        "transient", "Detect Transient Response", ConceptType.SIGNAL_FILTER,
        matched_primitive="detect_transient_response",
        inputs=[IOSpec(name="output", type_desc="ndarray"), IOSpec(name="n_transient_samples", type_desc="int")],
        outputs=[IOSpec(name="estimated_transient_length", type_desc="int"), IOSpec(name="transient_energy_fraction", type_desc="float")],
        description="Detect startup transient in filter output.",
        type_signature="ndarray, int -> tuple[int, float]",
    )
    rhs = CDGExport(nodes=[apply_f, transient, sink], edges=[_edge("apply_f", "transient"), _edge("transient", "sink")])

    return RewriteRule(
        name="insert_transient_detection_after_apply_filter", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"apply_f": "apply_f", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"apply_f": "apply_f", "sink": "sink"}, edge_map={}),
        priority=2,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_pole_stability(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    mag = intermediates.get("max_pole_magnitude")
    if mag is None:
        return None
    try:
        m = float(mag)
    except (ValueError, TypeError):
        return None
    if m >= 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_pole_stability_analysis_after_design",
            severity=1.0, evidence=f"Max pole magnitude {m:.4f} >= 1.0 — filter is unstable",
            metric_name="max_pole_magnitude", metric_value=m, threshold=1.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_passband_ripple(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ripple = intermediates.get("passband_ripple_db")
    if ripple is None:
        return None
    try:
        r = float(ripple)
    except (ValueError, TypeError):
        return None
    if r > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_passband_ripple_measurement_after_validate",
            severity=min(1.0, r / 3.0), evidence=f"Passband ripple {r:.2f} dB exceeds 1.0 dB",
            metric_name="passband_ripple_db", metric_value=r, threshold=1.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_group_delay(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    var = intermediates.get("group_delay_variation")
    if var is None:
        return None
    try:
        v = float(var)
    except (ValueError, TypeError):
        return None
    if v > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_group_delay_analysis_after_frequency_response",
            severity=min(1.0, v / 10.0), evidence=f"Group delay variation {v:.2f} samples exceeds 1.0 — phase distortion",
            metric_name="group_delay_variation", metric_value=v, threshold=1.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_transient(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("transient_energy_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_transient_detection_after_apply_filter",
            severity=min(1.0, f * 3), evidence=f"Transient energy fraction {f:.3f} exceeds 0.1 — startup artifacts",
            metric_name="transient_energy_fraction", metric_value=f, threshold=0.1, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SignalFilterExpansionRuleSet:
    """Expansion rules for signal filter pipelines (Butterworth, Chebyshev, FIR, Notch)."""

    name = "signal_filter"
    domain = "signal_filter"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_pole_stability_analysis(),
            _build_insert_passband_ripple_measurement(),
            _build_insert_group_delay_analysis(),
            _build_insert_transient_detection(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_pole_stability, _diagnose_passband_ripple, _diagnose_group_delay, _diagnose_transient]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

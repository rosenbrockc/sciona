"""Expansion rules for the Kalman Filter family (KF, EKF, UKF).

Kalman Filter skeleton topology (6 nodes, bipartite predict/update):

    Predict State ──→ Innovation ──→ Update State
    Predict Covariance → Kalman Gain → Update Covariance

Expansion insertion points:
  - After Innovation: innovation consistency check
  - After Predict Covariance: covariance PD validation
  - After Kalman Gain: gain magnitude analysis
  - After Update State: state smoothness check
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

_DOMAIN = "kalman_filter"

_PREDICT_STATE = "Predict State"
_PREDICT_COVARIANCE = "Predict Covariance"
_INNOVATION = "Innovation"
_KALMAN_GAIN = "Kalman Gain"
_UPDATE_STATE = "Update State"
_UPDATE_COVARIANCE = "Update Covariance"


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


def _build_insert_innovation_consistency_check() -> RewriteRule:
    innovation = _node("innovation", _INNOVATION, ConceptType.SEQUENTIAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[innovation, sink], edges=[_edge("innovation", "sink")])
    interface = CDGExport(nodes=[innovation, sink], edges=[])

    nis = _node(
        "nis", "Check Innovation Consistency", ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="check_innovation_consistency",
        inputs=[IOSpec(name="innovations", type_desc="ndarray"), IOSpec(name="innovation_covariance", type_desc="ndarray")],
        outputs=[IOSpec(name="mean_nis", type_desc="float"), IOSpec(name="is_consistent", type_desc="bool")],
        description="Check whether innovations are consistent with their predicted covariance.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[innovation, nis, sink], edges=[_edge("innovation", "nis"), _edge("nis", "sink")])

    return RewriteRule(
        name="insert_innovation_consistency_check_after_innovation", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"innovation": "innovation", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"innovation": "innovation", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_covariance_pd_validation() -> RewriteRule:
    pred_cov = _node("pred_cov", _PREDICT_COVARIANCE, ConceptType.SEQUENTIAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[pred_cov, sink], edges=[_edge("pred_cov", "sink")])
    interface = CDGExport(nodes=[pred_cov, sink], edges=[])

    pd_check = _node(
        "pd_check", "Validate Covariance PD", ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="validate_covariance_pd",
        inputs=[IOSpec(name="covariance", type_desc="ndarray")],
        outputs=[IOSpec(name="min_eigenvalue", type_desc="float"), IOSpec(name="is_pd", type_desc="bool")],
        description="Validate that a covariance matrix is positive definite.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[pred_cov, pd_check, sink], edges=[_edge("pred_cov", "pd_check"), _edge("pd_check", "sink")])

    return RewriteRule(
        name="insert_covariance_pd_validation_after_predict_covariance", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"pred_cov": "pred_cov", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"pred_cov": "pred_cov", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_gain_magnitude_analysis() -> RewriteRule:
    gain = _node("gain", _KALMAN_GAIN, ConceptType.CONJUGATE_UPDATE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[gain, sink], edges=[_edge("gain", "sink")])
    interface = CDGExport(nodes=[gain, sink], edges=[])

    mag = _node(
        "mag", "Analyze Kalman Gain Magnitude", ConceptType.CONJUGATE_UPDATE,
        matched_primitive="analyze_kalman_gain_magnitude",
        inputs=[IOSpec(name="kalman_gains", type_desc="ndarray")],
        outputs=[IOSpec(name="max_gain_norm", type_desc="float"), IOSpec(name="is_bounded", type_desc="bool")],
        description="Analyze the magnitude of Kalman gains over time.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[gain, mag, sink], edges=[_edge("gain", "mag"), _edge("mag", "sink")])

    return RewriteRule(
        name="insert_gain_magnitude_analysis_after_kalman_gain", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"gain": "gain", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"gain": "gain", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_state_smoothness_check() -> RewriteRule:
    update = _node("update", _UPDATE_STATE, ConceptType.CONJUGATE_UPDATE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    smooth = _node(
        "smooth", "Check State Smoothness", ConceptType.CONJUGATE_UPDATE,
        matched_primitive="check_state_smoothness",
        inputs=[IOSpec(name="state_estimates", type_desc="ndarray"), IOSpec(name="max_jump_ratio", type_desc="float")],
        outputs=[IOSpec(name="n_jumps", type_desc="int"), IOSpec(name="jump_fraction", type_desc="float")],
        description="Check for sudden jumps in state estimates.",
        type_signature="ndarray, float -> tuple[int, float]",
    )
    rhs = CDGExport(nodes=[update, smooth, sink], edges=[_edge("update", "smooth"), _edge("smooth", "sink")])

    return RewriteRule(
        name="insert_state_smoothness_check_after_update_state", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_innovation_consistency(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    nis = intermediates.get("mean_nis")
    dim = intermediates.get("innovation_dim", 1)
    if nis is None:
        return None
    try:
        n = float(nis)
        d = int(dim)
    except (ValueError, TypeError):
        return None
    if n > 2.0 * d or n < 0.5 * d:
        return ExpansionDiagnostic(
            rule_name="insert_innovation_consistency_check_after_innovation",
            severity=min(1.0, abs(n - d) / (2.0 * d)),
            evidence=f"Mean NIS {n:.2f} outside [{0.5*d:.1f}, {2.0*d:.1f}] — model mismatch",
            metric_name="mean_nis", metric_value=n, threshold=float(d), source_domain=_DOMAIN,
        )
    return None


def _diagnose_covariance_pd(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    min_eig = intermediates.get("min_covariance_eigenvalue")
    if min_eig is None:
        return None
    try:
        e = float(min_eig)
    except (ValueError, TypeError):
        return None
    if e <= 0:
        return ExpansionDiagnostic(
            rule_name="insert_covariance_pd_validation_after_predict_covariance",
            severity=1.0, evidence=f"Min covariance eigenvalue {e:.2e} <= 0 — not positive definite",
            metric_name="min_covariance_eigenvalue", metric_value=e, threshold=0.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_gain_magnitude(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    mag = intermediates.get("max_kalman_gain_norm")
    if mag is None:
        return None
    try:
        m = float(mag)
    except (ValueError, TypeError):
        return None
    if m > 100.0:
        return ExpansionDiagnostic(
            rule_name="insert_gain_magnitude_analysis_after_kalman_gain",
            severity=min(1.0, m / 1000.0), evidence=f"Max Kalman gain norm {m:.1f} exceeds 100 — potential divergence",
            metric_name="max_kalman_gain_norm", metric_value=m, threshold=100.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_state_smoothness(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("state_jump_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.05:
        return ExpansionDiagnostic(
            rule_name="insert_state_smoothness_check_after_update_state",
            severity=min(1.0, f * 10), evidence=f"State jump fraction {f:.3f} exceeds 0.05 — outlier measurements",
            metric_name="state_jump_fraction", metric_value=f, threshold=0.05, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class KalmanFilterExpansionRuleSet:
    """Expansion rules for Kalman filter pipelines (KF, EKF, UKF)."""

    name = "kalman_filter"
    domain = "kalman_filter"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_innovation_consistency_check(),
            _build_insert_covariance_pd_validation(),
            _build_insert_gain_magnitude_analysis(),
            _build_insert_state_smoothness_check(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_innovation_consistency, _diagnose_covariance_pd, _diagnose_gain_magnitude, _diagnose_state_smoothness]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

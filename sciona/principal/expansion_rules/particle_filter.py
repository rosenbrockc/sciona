"""Expansion rules for the Particle Filter family.

Particle Filter skeleton topology (4 nodes, linear with fan-in):

    Preprocess → Predict → Reweight → Postprocess

Expansion insertion points:
  - After Reweight: effective sample size monitoring
  - After Predict: particle diversity analysis
  - After Postprocess: weight variance tracking
  - After Preprocess: resampling quality check
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

_DOMAIN = "particle_filter"

_PREPROCESS = "Preprocess"
_PREDICT = "Predict"
_REWEIGHT = "Reweight"
_POSTPROCESS = "Postprocess"


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


def _build_insert_ess_monitoring() -> RewriteRule:
    reweight = _node("reweight", _REWEIGHT, ConceptType.PROBABILISTIC_ORACLE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[reweight, sink], edges=[_edge("reweight", "sink")])
    interface = CDGExport(nodes=[reweight, sink], edges=[])

    ess = _node(
        "ess", "Monitor Effective Sample Size", ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="monitor_effective_sample_size",
        inputs=[IOSpec(name="log_weights", type_desc="ndarray")],
        outputs=[IOSpec(name="ess_fraction", type_desc="float"), IOSpec(name="is_healthy", type_desc="bool")],
        description="Monitor the effective sample size (ESS) of particle weights.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[reweight, ess, sink], edges=[_edge("reweight", "ess"), _edge("ess", "sink")])

    return RewriteRule(
        name="insert_ess_monitoring_after_reweight", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"reweight": "reweight", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"reweight": "reweight", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_particle_diversity_analysis() -> RewriteRule:
    predict = _node("predict", _PREDICT, ConceptType.SEQUENTIAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[predict, sink], edges=[_edge("predict", "sink")])
    interface = CDGExport(nodes=[predict, sink], edges=[])

    diversity = _node(
        "diversity", "Analyze Particle Diversity", ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="analyze_particle_diversity",
        inputs=[IOSpec(name="particles", type_desc="ndarray")],
        outputs=[IOSpec(name="mean_pairwise_distance", type_desc="float"), IOSpec(name="is_diverse", type_desc="bool")],
        description="Analyze diversity of particle positions.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[predict, diversity, sink], edges=[_edge("predict", "diversity"), _edge("diversity", "sink")])

    return RewriteRule(
        name="insert_particle_diversity_analysis_after_predict", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"predict": "predict", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"predict": "predict", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_weight_variance_tracking() -> RewriteRule:
    postprocess = _node("postprocess", _POSTPROCESS, ConceptType.SEQUENTIAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[postprocess, sink], edges=[_edge("postprocess", "sink")])
    interface = CDGExport(nodes=[postprocess, sink], edges=[])

    wvar = _node(
        "wvar", "Track Weight Variance", ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="track_weight_variance",
        inputs=[IOSpec(name="log_weights_history", type_desc="ndarray")],
        outputs=[IOSpec(name="variance_trend", type_desc="float"), IOSpec(name="is_stable", type_desc="bool")],
        description="Track variance of particle weights over time.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[postprocess, wvar, sink], edges=[_edge("postprocess", "wvar"), _edge("wvar", "sink")])

    return RewriteRule(
        name="insert_weight_variance_tracking_after_postprocess", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"postprocess": "postprocess", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"postprocess": "postprocess", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _build_insert_resampling_quality_check() -> RewriteRule:
    preprocess = _node("preprocess", _PREPROCESS, ConceptType.SEQUENTIAL_FILTER)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[preprocess, sink], edges=[_edge("preprocess", "sink")])
    interface = CDGExport(nodes=[preprocess, sink], edges=[])

    resample = _node(
        "resample", "Check Resampling Quality", ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="check_resampling_quality",
        inputs=[IOSpec(name="parent_indices", type_desc="ndarray"), IOSpec(name="n_particles", type_desc="int")],
        outputs=[IOSpec(name="max_duplication_fraction", type_desc="float"), IOSpec(name="is_acceptable", type_desc="bool")],
        description="Check the quality of resampling by analyzing duplication.",
        type_signature="ndarray, int -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[preprocess, resample, sink], edges=[_edge("preprocess", "resample"), _edge("resample", "sink")])

    return RewriteRule(
        name="insert_resampling_quality_check_after_preprocess", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"preprocess": "preprocess", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"preprocess": "preprocess", "sink": "sink"}, edge_map={}),
        priority=2,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_ess(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ess = intermediates.get("ess_fraction")
    if ess is None:
        return None
    try:
        e = float(ess)
    except (ValueError, TypeError):
        return None
    if e < 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_ess_monitoring_after_reweight",
            severity=min(1.0, (0.5 - e) / 0.5), evidence=f"ESS fraction {e:.3f} below 0.5 — weight degeneracy",
            metric_name="ess_fraction", metric_value=e, threshold=0.5, source_domain=_DOMAIN,
        )
    return None


def _diagnose_diversity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    diverse = intermediates.get("particle_diversity_low")
    if diverse is None:
        return None
    try:
        d = bool(diverse)
    except (ValueError, TypeError):
        return None
    if d:
        return ExpansionDiagnostic(
            rule_name="insert_particle_diversity_analysis_after_predict",
            severity=0.7, evidence="Particle diversity is low — potential mode collapse",
            metric_name="particle_diversity_low", metric_value=1.0, threshold=0.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_weight_variance(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    trend = intermediates.get("weight_variance_trend")
    if trend is None:
        return None
    try:
        t = float(trend)
    except (ValueError, TypeError):
        return None
    if t > 0:
        return ExpansionDiagnostic(
            rule_name="insert_weight_variance_tracking_after_postprocess",
            severity=min(1.0, t * 10), evidence=f"Weight variance trend {t:.4f} positive — progressive degeneracy",
            metric_name="weight_variance_trend", metric_value=t, threshold=0.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_resampling(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("max_duplication_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_resampling_quality_check_after_preprocess",
            severity=min(1.0, f * 5), evidence=f"Max duplication fraction {f:.3f} exceeds 0.1 — aggressive resampling",
            metric_name="max_duplication_fraction", metric_value=f, threshold=0.1, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class ParticleFilterExpansionRuleSet:
    """Expansion rules for particle filter pipelines."""

    name = "particle_filter"
    domain = "particle_filter"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_ess_monitoring(),
            _build_insert_particle_diversity_analysis(),
            _build_insert_weight_variance_tracking(),
            _build_insert_resampling_quality_check(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_ess, _diagnose_diversity, _diagnose_weight_variance, _diagnose_resampling]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

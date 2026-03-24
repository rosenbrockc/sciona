"""Expansion rules for the Quadrature family."""

from __future__ import annotations

import logging

import numpy as np

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

logger = logging.getLogger(__name__)

_DOMAIN = "quadrature"

_SAMPLE_POINTS = "Sample Points"
_EVALUATE_INTEGRAND = "Evaluate Integrand"
_ESTIMATE_ERROR_REFINE = "Estimate Error/Refine"


def _node(node_id: str, name: str, concept_type: ConceptType, *, matched_primitive: str | None = None,
          inputs: list[IOSpec] | None = None, outputs: list[IOSpec] | None = None,
          description: str = "", type_signature: str = "") -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=description or name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=inputs or [],
        outputs=outputs or [],
        type_signature=type_signature or f"{name} -> result",
    )


def _edge(source_id: str, target_id: str, output_name: str = "out", input_name: str = "in",
          type_desc: str = "ndarray") -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=type_desc,
        target_type=type_desc,
    )


def _build_insert_domain_coverage_check() -> RewriteRule:
    sample = _node("sample", _SAMPLE_POINTS, ConceptType.QUADRATURE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[sample, sink], edges=[_edge("sample", "sink")])
    interface = CDGExport(nodes=[sample, sink], edges=[])
    coverage = _node(
        "coverage",
        "Check Domain Coverage",
        ConceptType.QUADRATURE,
        matched_primitive="check_domain_coverage",
        inputs=[IOSpec(name="points", type_desc="ndarray"), IOSpec(name="domain", type_desc="ndarray")],
        outputs=[IOSpec(name="max_gap_ratio", type_desc="float"), IOSpec(name="is_covered", type_desc="bool")],
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[sample, coverage, sink], edges=[_edge("sample", "coverage"), _edge("coverage", "sink")])
    return RewriteRule(
        name="insert_domain_coverage_check_after_sample",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"sample": "sample", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"sample": "sample", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_integrand_smoothness_analysis() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    evaluate = _node("evaluate", _EVALUATE_INTEGRAND, ConceptType.QUADRATURE)
    lhs = CDGExport(nodes=[src, evaluate], edges=[_edge("src", "evaluate")])
    interface = CDGExport(nodes=[src, evaluate], edges=[])
    smoothness = _node(
        "smoothness",
        "Analyze Integrand Smoothness",
        ConceptType.QUADRATURE,
        matched_primitive="analyze_integrand_smoothness",
        inputs=[IOSpec(name="values", type_desc="ndarray"), IOSpec(name="points", type_desc="ndarray")],
        outputs=[IOSpec(name="max_derivative", type_desc="float"), IOSpec(name="is_smooth", type_desc="bool")],
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[src, smoothness, evaluate], edges=[_edge("src", "smoothness"), _edge("smoothness", "evaluate")])
    return RewriteRule(
        name="insert_integrand_smoothness_analysis_before_evaluate",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "evaluate": "evaluate"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "evaluate": "evaluate"}, edge_map={}),
        priority=2,
    )


def _build_insert_singularity_detection() -> RewriteRule:
    evaluate = _node("evaluate", _EVALUATE_INTEGRAND, ConceptType.QUADRATURE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[evaluate, sink], edges=[_edge("evaluate", "sink")])
    interface = CDGExport(nodes=[evaluate, sink], edges=[])
    singular = _node(
        "singular",
        "Detect Singularity",
        ConceptType.QUADRATURE,
        matched_primitive="detect_singularity",
        inputs=[IOSpec(name="values", type_desc="ndarray")],
        outputs=[IOSpec(name="max_value", type_desc="float"), IOSpec(name="is_nonsingular", type_desc="bool")],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[evaluate, singular, sink], edges=[_edge("evaluate", "singular"), _edge("singular", "sink")])
    return RewriteRule(
        name="insert_singularity_detection_after_evaluate",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"evaluate": "evaluate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"evaluate": "evaluate", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_convergence_monitor() -> RewriteRule:
    refine = _node("refine", _ESTIMATE_ERROR_REFINE, ConceptType.QUADRATURE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[refine, sink], edges=[_edge("refine", "sink")])
    interface = CDGExport(nodes=[refine, sink], edges=[])
    convergence = _node(
        "convergence",
        "Monitor Convergence Rate",
        ConceptType.QUADRATURE,
        matched_primitive="monitor_convergence_rate",
        inputs=[IOSpec(name="estimates", type_desc="ndarray")],
        outputs=[IOSpec(name="rate", type_desc="float"), IOSpec(name="is_converging", type_desc="bool")],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[refine, convergence, sink], edges=[_edge("refine", "convergence"), _edge("convergence", "sink")])
    return RewriteRule(
        name="insert_convergence_monitor_after_refine",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"refine": "refine", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"refine": "refine", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _diagnose_coverage(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("max_gap_ratio")
    if value is None:
        return None
    try:
        gap = float(value)
    except (ValueError, TypeError):
        return None
    if gap > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_domain_coverage_check_after_sample",
            severity=max(0.35, min(1.0, gap)),
            evidence=f"Maximum domain gap ratio {gap:.3f} exceeds 0.1.",
            metric_name="max_gap_ratio",
            metric_value=gap,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_smoothness(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("integrand_max_derivative")
    if value is None:
        return None
    try:
        deriv = float(value)
    except (ValueError, TypeError):
        return None
    if deriv > 1e6:
        return ExpansionDiagnostic(
            rule_name="insert_integrand_smoothness_analysis_before_evaluate",
            severity=max(0.35, min(1.0, np.log10(max(deriv, 1.0)) / 8.0)),
            evidence=f"Maximum derivative {deriv:.2e} exceeds 1e6.",
            metric_name="integrand_max_derivative",
            metric_value=deriv,
            threshold=1e6,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_singularity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("integrand_max_value")
    if value is None:
        return None
    try:
        maximum = float(value)
    except (ValueError, TypeError):
        return None
    if maximum > 1e10:
        return ExpansionDiagnostic(
            rule_name="insert_singularity_detection_after_evaluate",
            severity=max(0.35, min(1.0, np.log10(max(maximum, 1.0)) / 12.0)),
            evidence=f"Maximum integrand value {maximum:.2e} exceeds 1e10.",
            metric_name="integrand_max_value",
            metric_value=maximum,
            threshold=1e10,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_convergence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("quadrature_convergence_rate")
    if value is None:
        return None
    try:
        rate = float(value)
    except (ValueError, TypeError):
        return None
    if rate >= 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_convergence_monitor_after_refine",
            severity=max(0.35, min(1.0, rate)),
            evidence=f"Quadrature convergence rate {rate:.3f} is too slow.",
            metric_name="quadrature_convergence_rate",
            metric_value=rate,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


class QuadratureExpansionRuleSet:
    @property
    def name(self) -> str:
        return "quadrature"

    @property
    def domain(self) -> str:
        return _DOMAIN

    def rules(self) -> list[RewriteRule]:
        return [
            _build_insert_domain_coverage_check(),
            _build_insert_integrand_smoothness_analysis(),
            _build_insert_singularity_detection(),
            _build_insert_convergence_monitor(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics = [
            _diagnose_coverage(cdg, context),
            _diagnose_smoothness(cdg, context),
            _diagnose_singularity(cdg, context),
            _diagnose_convergence(cdg, context),
        ]
        return [d for d in diagnostics if d is not None]

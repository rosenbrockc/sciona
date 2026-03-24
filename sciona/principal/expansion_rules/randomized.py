"""Expansion rules for the Randomized Algorithms family."""

from __future__ import annotations

import logging

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

logger = logging.getLogger(__name__)

_DOMAIN = "randomized"

_GENERATE_SAMPLES = "Generate Samples"
_SKETCH_HASH = "Sketch/Hash"
_ESTIMATE = "Estimate"


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


def _build_insert_sample_coverage_monitor() -> RewriteRule:
    generate = _node("generate", _GENERATE_SAMPLES, ConceptType.RANDOMIZED)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[generate, sink], edges=[_edge("generate", "sink")])
    interface = CDGExport(nodes=[generate, sink], edges=[])
    coverage = _node(
        "coverage",
        "Monitor Sample Coverage",
        ConceptType.RANDOMIZED,
        matched_primitive="monitor_sample_coverage",
        inputs=[IOSpec(name="samples", type_desc="ndarray"), IOSpec(name="population_size", type_desc="int")],
        outputs=[IOSpec(name="coverage", type_desc="float"), IOSpec(name="is_sufficient", type_desc="bool")],
        type_signature="ndarray, int -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[generate, coverage, sink], edges=[_edge("generate", "coverage"), _edge("coverage", "sink")])
    return RewriteRule(
        name="insert_sample_coverage_monitor_after_generate",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"generate": "generate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"generate": "generate", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_hash_independence_validation() -> RewriteRule:
    sketch = _node("sketch", _SKETCH_HASH, ConceptType.RANDOMIZED)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[sketch, sink], edges=[_edge("sketch", "sink")])
    interface = CDGExport(nodes=[sketch, sink], edges=[])
    independence = _node(
        "independence",
        "Validate Hash Independence",
        ConceptType.RANDOMIZED,
        matched_primitive="validate_hash_independence",
        inputs=[
            IOSpec(name="observed_collisions", type_desc="float"),
            IOSpec(name="expected_collisions", type_desc="float"),
        ],
        outputs=[IOSpec(name="collision_ratio", type_desc="float"), IOSpec(name="is_independent", type_desc="bool")],
        type_signature="float, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[sketch, independence, sink], edges=[_edge("sketch", "independence"), _edge("independence", "sink")])
    return RewriteRule(
        name="insert_hash_independence_validation_after_sketch",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"sketch": "sketch", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"sketch": "sketch", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_sketch_accuracy_analysis() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    estimate = _node("estimate", _ESTIMATE, ConceptType.RANDOMIZED)
    lhs = CDGExport(nodes=[src, estimate], edges=[_edge("src", "estimate")])
    interface = CDGExport(nodes=[src, estimate], edges=[])
    accuracy = _node(
        "accuracy",
        "Analyze Sketch Accuracy",
        ConceptType.RANDOMIZED,
        matched_primitive="analyze_sketch_accuracy",
        inputs=[IOSpec(name="true_values", type_desc="ndarray"), IOSpec(name="estimated_values", type_desc="ndarray")],
        outputs=[IOSpec(name="relative_error", type_desc="float"), IOSpec(name="is_accurate", type_desc="bool")],
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[src, accuracy, estimate], edges=[_edge("src", "accuracy"), _edge("accuracy", "estimate")])
    return RewriteRule(
        name="insert_sketch_accuracy_analysis_before_estimate",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "estimate": "estimate"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "estimate": "estimate"}, edge_map={}),
        priority=2,
    )


def _build_insert_concentration_check() -> RewriteRule:
    estimate = _node("estimate", _ESTIMATE, ConceptType.RANDOMIZED)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[estimate, sink], edges=[_edge("estimate", "sink")])
    interface = CDGExport(nodes=[estimate, sink], edges=[])
    concentration = _node(
        "concentration",
        "Check Concentration Bound",
        ConceptType.RANDOMIZED,
        matched_primitive="check_concentration_bound",
        inputs=[IOSpec(name="empirical_errors", type_desc="ndarray"), IOSpec(name="theoretical_bound", type_desc="float")],
        outputs=[IOSpec(name="violation_rate", type_desc="float"), IOSpec(name="is_within_bound", type_desc="bool")],
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[estimate, concentration, sink], edges=[_edge("estimate", "concentration"), _edge("concentration", "sink")])
    return RewriteRule(
        name="insert_concentration_check_after_estimate",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"estimate": "estimate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"estimate": "estimate", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _diagnose_coverage(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("sample_coverage")
    if value is None:
        return None
    try:
        coverage = float(value)
    except (ValueError, TypeError):
        return None
    if coverage < 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_sample_coverage_monitor_after_generate",
            severity=max(0.35, min(1.0, (0.1 - coverage) / 0.1)),
            evidence=f"Sample coverage {coverage:.3f} is below 0.1.",
            metric_name="sample_coverage",
            metric_value=coverage,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_hash_independence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("collision_ratio")
    if value is None:
        return None
    try:
        ratio = float(value)
    except (ValueError, TypeError):
        return None
    if ratio > 2.0:
        return ExpansionDiagnostic(
            rule_name="insert_hash_independence_validation_after_sketch",
            severity=max(0.35, min(1.0, ratio / 4.0)),
            evidence=f"Collision ratio {ratio:.3f} exceeds 2.0.",
            metric_name="collision_ratio",
            metric_value=ratio,
            threshold=2.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_sketch_accuracy(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("sketch_relative_error")
    if value is None:
        return None
    try:
        err = float(value)
    except (ValueError, TypeError):
        return None
    if err > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_sketch_accuracy_analysis_before_estimate",
            severity=max(0.35, min(1.0, err)),
            evidence=f"Sketch relative error {err:.3f} exceeds 0.1.",
            metric_name="sketch_relative_error",
            metric_value=err,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_concentration(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("concentration_violation_rate")
    if value is None:
        return None
    try:
        rate = float(value)
    except (ValueError, TypeError):
        return None
    if rate > 0.05:
        return ExpansionDiagnostic(
            rule_name="insert_concentration_check_after_estimate",
            severity=max(0.35, min(1.0, rate / 0.25)),
            evidence=f"Concentration violation rate {rate:.3f} exceeds 0.05.",
            metric_name="concentration_violation_rate",
            metric_value=rate,
            threshold=0.05,
            source_domain=_DOMAIN,
        )
    return None


class RandomizedExpansionRuleSet:
    @property
    def name(self) -> str:
        return "randomized"

    @property
    def domain(self) -> str:
        return _DOMAIN

    def rules(self) -> list[RewriteRule]:
        return [
            _build_insert_sample_coverage_monitor(),
            _build_insert_hash_independence_validation(),
            _build_insert_sketch_accuracy_analysis(),
            _build_insert_concentration_check(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics = [
            _diagnose_coverage(cdg, context),
            _diagnose_hash_independence(cdg, context),
            _diagnose_sketch_accuracy(cdg, context),
            _diagnose_concentration(cdg, context),
        ]
        return [d for d in diagnostics if d is not None]

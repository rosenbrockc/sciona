"""Expansion rules for the Information Theory family."""

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
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

logger = logging.getLogger(__name__)

_DOMAIN = "information_theory"

_ESTIMATE_DISTRIBUTION = "Estimate Distribution"
_COMPUTE_ENTROPY_DIVERGENCE = "Compute Entropy/Divergence"
_VALIDATE_BOUNDS = "Validate Bounds"


def _node(
    node_id: str,
    name: str,
    concept_type: ConceptType,
    *,
    matched_primitive: str | None = None,
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
    description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
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


def _edge(
    source_id: str,
    target_id: str,
    output_name: str = "out",
    input_name: str = "in",
    type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=type_desc,
        target_type=type_desc,
    )


def _build_insert_distribution_support_check() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    compute = _node("compute", _COMPUTE_ENTROPY_DIVERGENCE, ConceptType.INFORMATION_THEORY)
    lhs = CDGExport(nodes=[src, compute], edges=[_edge("src", "compute")])
    interface = CDGExport(nodes=[src, compute], edges=[])

    support = _node(
        "support",
        "Check Distribution Support",
        ConceptType.INFORMATION_THEORY,
        matched_primitive="check_distribution_support",
        inputs=[IOSpec(name="probabilities", type_desc="ndarray")],
        outputs=[
            IOSpec(name="zero_mass_fraction", type_desc="float"),
            IOSpec(name="has_full_support", type_desc="bool"),
        ],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, support, compute],
        edges=[_edge("src", "support"), _edge("support", "compute")],
    )
    return RewriteRule(
        name="insert_distribution_support_check_before_compute",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "compute": "compute"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "compute": "compute"}, edge_map={}),
        priority=3,
    )


def _build_insert_sample_sufficiency_analysis() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    compute = _node("compute", _COMPUTE_ENTROPY_DIVERGENCE, ConceptType.INFORMATION_THEORY)
    lhs = CDGExport(nodes=[src, compute], edges=[_edge("src", "compute")])
    interface = CDGExport(nodes=[src, compute], edges=[])

    sufficiency = _node(
        "sufficiency",
        "Analyze Sample Sufficiency",
        ConceptType.INFORMATION_THEORY,
        matched_primitive="analyze_sample_sufficiency",
        inputs=[
            IOSpec(name="sample_count", type_desc="int"),
            IOSpec(name="support_size", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="samples_per_symbol", type_desc="float"),
            IOSpec(name="is_sufficient", type_desc="bool"),
        ],
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, sufficiency, compute],
        edges=[_edge("src", "sufficiency"), _edge("sufficiency", "compute")],
    )
    return RewriteRule(
        name="insert_sample_sufficiency_analysis_before_compute",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "compute": "compute"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "compute": "compute"}, edge_map={}),
        priority=2,
    )


def _build_insert_numerical_underflow_detection() -> RewriteRule:
    compute = _node("compute", _COMPUTE_ENTROPY_DIVERGENCE, ConceptType.INFORMATION_THEORY)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[compute, sink], edges=[_edge("compute", "sink")])
    interface = CDGExport(nodes=[compute, sink], edges=[])

    underflow = _node(
        "underflow",
        "Detect Numerical Underflow",
        ConceptType.INFORMATION_THEORY,
        matched_primitive="detect_numerical_underflow",
        inputs=[IOSpec(name="log_probabilities", type_desc="ndarray")],
        outputs=[
            IOSpec(name="underflow_fraction", type_desc="float"),
            IOSpec(name="is_stable", type_desc="bool"),
        ],
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[compute, underflow, sink],
        edges=[_edge("compute", "underflow"), _edge("underflow", "sink")],
    )
    return RewriteRule(
        name="insert_numerical_underflow_detection_after_compute",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"compute": "compute", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"compute": "compute", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_information_inequality_validation() -> RewriteRule:
    validate = _node("validate", _VALIDATE_BOUNDS, ConceptType.INFORMATION_THEORY)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[validate, sink], edges=[_edge("validate", "sink")])
    interface = CDGExport(nodes=[validate, sink], edges=[])

    inequality = _node(
        "inequality",
        "Validate Information Inequality",
        ConceptType.INFORMATION_THEORY,
        matched_primitive="validate_information_inequality",
        inputs=[
            IOSpec(name="lhs_values", type_desc="ndarray"),
            IOSpec(name="rhs_values", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="max_violation", type_desc="float"),
            IOSpec(name="inequality_holds", type_desc="bool"),
        ],
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[validate, inequality, sink],
        edges=[_edge("validate", "inequality"), _edge("inequality", "sink")],
    )
    return RewriteRule(
        name="insert_information_inequality_validation_after_validate",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"validate": "validate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"validate": "validate", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _diagnose_distribution_support(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("zero_mass_fraction")
    if value is None:
        return None
    try:
        frac = float(value)
    except (ValueError, TypeError):
        return None
    if frac > 0.0:
        return ExpansionDiagnostic(
            rule_name="insert_distribution_support_check_before_compute",
            severity=max(0.35, min(1.0, frac)),
            evidence=f"Zero-mass fraction {frac:.3f} indicates missing support.",
            metric_name="zero_mass_fraction",
            metric_value=frac,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_sample_sufficiency(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("samples_per_symbol")
    if value is None:
        return None
    try:
        sps = float(value)
    except (ValueError, TypeError):
        return None
    if sps < 5.0:
        return ExpansionDiagnostic(
            rule_name="insert_sample_sufficiency_analysis_before_compute",
            severity=max(0.35, min(1.0, (5.0 - sps) / 5.0)),
            evidence=f"Samples per symbol {sps:.3f} is below 5.0.",
            metric_name="samples_per_symbol",
            metric_value=sps,
            threshold=5.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_numerical_underflow(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("underflow_fraction")
    if value is None:
        return None
    try:
        frac = float(value)
    except (ValueError, TypeError):
        return None
    if frac > 0.05:
        return ExpansionDiagnostic(
            rule_name="insert_numerical_underflow_detection_after_compute",
            severity=max(0.35, min(1.0, frac)),
            evidence=f"Underflow fraction {frac:.3f} exceeds 0.05.",
            metric_name="underflow_fraction",
            metric_value=frac,
            threshold=0.05,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_information_inequality(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    value = (context.intermediates or {}).get("max_information_inequality_violation")
    if value is None:
        value = (context.intermediates or {}).get("max_violation")
    if value is None:
        return None
    try:
        violation = float(value)
    except (ValueError, TypeError):
        return None
    if violation > 1e-9:
        return ExpansionDiagnostic(
            rule_name="insert_information_inequality_validation_after_validate",
            severity=max(0.35, min(1.0, violation / 1e-6)),
            evidence=f"Maximum information-inequality violation {violation:.2e} exceeds 1e-9.",
            metric_name="max_information_inequality_violation",
            metric_value=violation,
            threshold=1e-9,
            source_domain=_DOMAIN,
        )
    return None


class InformationTheoryExpansionRuleSet:
    """Expansion rules for information-theoretic pipelines."""

    name = "information_theory"
    domain = "information_theory"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_distribution_support_check(),
            _build_insert_sample_sufficiency_analysis(),
            _build_insert_numerical_underflow_detection(),
            _build_insert_information_inequality_validation(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in (
            _diagnose_distribution_support,
            _diagnose_sample_sufficiency,
            _diagnose_numerical_underflow,
            _diagnose_information_inequality,
        ):
            diagnostic = fn(cdg, context)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

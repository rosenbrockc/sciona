"""Expansion rules for the Geometry family (Convex Hull, Closest Pair, Segment Intersection).

Geometry skeleton topology (3 nodes, linear pipeline):

    Preprocess Points → Construct → Verify Invariant

Expansion insertion points:
  - Before Preprocess Points: collinearity detection, duplicate point detection
  - After Construct: numeric precision analysis
  - After Verify Invariant: convexity validation
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

_DOMAIN = "geometry"

_PREPROCESS_POINTS = "Preprocess Points"
_CONSTRUCT = "Construct"
_VERIFY_INVARIANT = "Verify Invariant"


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


def _build_insert_collinearity_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    preprocess = _node("preprocess", _PREPROCESS_POINTS, ConceptType.GEOMETRY)
    lhs = CDGExport(nodes=[src, preprocess], edges=[_edge("src", "preprocess")])
    interface = CDGExport(nodes=[src, preprocess], edges=[])

    collinear = _node(
        "collinear", "Detect Collinear Points", ConceptType.GEOMETRY,
        matched_primitive="detect_collinear_points",
        inputs=[IOSpec(name="points", type_desc="ndarray"), IOSpec(name="tolerance", type_desc="float")],
        outputs=[IOSpec(name="n_collinear_triples", type_desc="int"), IOSpec(name="collinear_fraction", type_desc="float")],
        description="Detect collinear point triples that cause degenerate geometry.",
        type_signature="ndarray, float -> tuple[int, float]",
    )
    rhs = CDGExport(nodes=[src, collinear, preprocess], edges=[_edge("src", "collinear"), _edge("collinear", "preprocess")])

    return RewriteRule(
        name="insert_collinearity_detection_before_preprocess", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        priority=3,
    )


def _build_insert_duplicate_point_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    preprocess = _node("preprocess", _PREPROCESS_POINTS, ConceptType.GEOMETRY)
    lhs = CDGExport(nodes=[src, preprocess], edges=[_edge("src", "preprocess")])
    interface = CDGExport(nodes=[src, preprocess], edges=[])

    dup = _node(
        "dup", "Detect Duplicate Points", ConceptType.GEOMETRY,
        matched_primitive="detect_duplicate_points",
        inputs=[IOSpec(name="points", type_desc="ndarray"), IOSpec(name="tolerance", type_desc="float")],
        outputs=[IOSpec(name="n_duplicates", type_desc="int"), IOSpec(name="duplicate_fraction", type_desc="float")],
        description="Detect duplicate or near-duplicate points.",
        type_signature="ndarray, float -> tuple[int, float]",
    )
    rhs = CDGExport(nodes=[src, dup, preprocess], edges=[_edge("src", "dup"), _edge("dup", "preprocess")])

    return RewriteRule(
        name="insert_duplicate_point_detection_before_preprocess", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "preprocess": "preprocess"}, edge_map={}),
        priority=2,
    )


def _build_insert_numeric_precision_analysis() -> RewriteRule:
    construct = _node("construct", _CONSTRUCT, ConceptType.GEOMETRY)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[construct, sink], edges=[_edge("construct", "sink")])
    interface = CDGExport(nodes=[construct, sink], edges=[])

    precision = _node(
        "precision", "Analyze Numeric Precision", ConceptType.GEOMETRY,
        matched_primitive="analyze_numeric_precision",
        inputs=[IOSpec(name="points", type_desc="ndarray")],
        outputs=[IOSpec(name="condition_number", type_desc="float"), IOSpec(name="is_risky", type_desc="bool")],
        description="Analyze whether point coordinates risk floating-point issues.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[construct, precision, sink], edges=[_edge("construct", "precision"), _edge("precision", "sink")])

    return RewriteRule(
        name="insert_numeric_precision_analysis_after_construct", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"construct": "construct", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"construct": "construct", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_convexity_validation() -> RewriteRule:
    verify = _node("verify", _VERIFY_INVARIANT, ConceptType.GEOMETRY)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[verify, sink], edges=[_edge("verify", "sink")])
    interface = CDGExport(nodes=[verify, sink], edges=[])

    convexity = _node(
        "convexity", "Validate Convexity", ConceptType.GEOMETRY,
        matched_primitive="validate_convexity",
        inputs=[IOSpec(name="hull_points", type_desc="ndarray")],
        outputs=[IOSpec(name="n_violations", type_desc="int"), IOSpec(name="is_convex", type_desc="bool")],
        description="Validate that a polygon is convex by checking cross-product signs.",
        type_signature="ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(nodes=[verify, convexity, sink], edges=[_edge("verify", "convexity"), _edge("convexity", "sink")])

    return RewriteRule(
        name="insert_convexity_validation_after_verify", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"verify": "verify", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"verify": "verify", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_collinearity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("collinear_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_collinearity_detection_before_preprocess",
            severity=min(1.0, f), evidence=f"Collinear fraction {f:.2f} exceeds 0.1 — degenerate geometry risk",
            metric_name="collinear_fraction", metric_value=f, threshold=0.1, source_domain=_DOMAIN,
        )
    return None


def _diagnose_duplicates(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("duplicate_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.01:
        return ExpansionDiagnostic(
            rule_name="insert_duplicate_point_detection_before_preprocess",
            severity=min(1.0, f * 10), evidence=f"Duplicate fraction {f:.3f} exceeds 0.01 — may cause degenerate output",
            metric_name="duplicate_fraction", metric_value=f, threshold=0.01, source_domain=_DOMAIN,
        )
    return None


def _diagnose_numeric_precision(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    cond = intermediates.get("geometric_condition_number")
    if cond is None:
        return None
    try:
        c = float(cond)
    except (ValueError, TypeError):
        return None
    if c > 1e10:
        return ExpansionDiagnostic(
            rule_name="insert_numeric_precision_analysis_after_construct",
            severity=min(1.0, c / 1e15), evidence=f"Condition number {c:.2e} exceeds 1e10 — floating-point predicates may fail",
            metric_name="geometric_condition_number", metric_value=c, threshold=1e10, source_domain=_DOMAIN,
        )
    return None


def _diagnose_convexity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    violations = intermediates.get("convexity_violations")
    if violations is None:
        return None
    try:
        v = int(violations)
    except (ValueError, TypeError):
        return None
    if v > 0:
        return ExpansionDiagnostic(
            rule_name="insert_convexity_validation_after_verify",
            severity=min(1.0, v / 5.0), evidence=f"{v} convexity violation(s) — output polygon is not convex",
            metric_name="convexity_violations", metric_value=float(v), threshold=0.0, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class GeometryExpansionRuleSet:
    """Expansion rules for geometry pipelines (Convex Hull, Closest Pair, Segment Intersection)."""

    name = "geometry"
    domain = "geometry"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_collinearity_detection(),
            _build_insert_duplicate_point_detection(),
            _build_insert_numeric_precision_analysis(),
            _build_insert_convexity_validation(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_collinearity, _diagnose_duplicates, _diagnose_numeric_precision, _diagnose_convexity]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

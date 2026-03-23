"""Expansion rules for the Number Theory family (Euclidean GCD, Prime Sieve, Modular Exponentiation).

Number Theory skeleton topology (3 nodes, linear pipeline):

    Reduce → Iterate → Conclude

Expansion insertion points:
  - Before Reduce: input range validation, modular overflow detection
  - After Iterate: GCD convergence monitoring
  - Before Conclude: small prime divisor check
"""

from __future__ import annotations

import logging

import numpy as np

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

_DOMAIN = "number_theory"

_REDUCE = "Reduce"
_ITERATE = "Iterate"
_CONCLUDE = "Conclude"


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


def _build_insert_input_range_validation() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    reduce = _node("reduce", _REDUCE, ConceptType.NUMBER_THEORY)
    lhs = CDGExport(nodes=[src, reduce], edges=[_edge("src", "reduce")])
    interface = CDGExport(nodes=[src, reduce], edges=[])

    validate = _node(
        "validate", "Validate Input Range", ConceptType.NUMBER_THEORY,
        matched_primitive="validate_input_range",
        inputs=[IOSpec(name="values", type_desc="ndarray"), IOSpec(name="bit_width", type_desc="int")],
        outputs=[IOSpec(name="n_overflow_risk", type_desc="int"), IOSpec(name="all_safe", type_desc="bool")],
        description="Check whether input values are within safe computation range.",
        type_signature="ndarray, int -> tuple[int, bool]",
    )
    rhs = CDGExport(nodes=[src, validate, reduce], edges=[_edge("src", "validate"), _edge("validate", "reduce")])

    return RewriteRule(
        name="insert_input_range_validation_before_reduce", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "reduce": "reduce"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "reduce": "reduce"}, edge_map={}),
        priority=3,
    )


def _build_insert_gcd_convergence_monitoring() -> RewriteRule:
    iterate = _node("iterate", _ITERATE, ConceptType.NUMBER_THEORY)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[iterate, sink], edges=[_edge("iterate", "sink")])
    interface = CDGExport(nodes=[iterate, sink], edges=[])

    convergence = _node(
        "convergence", "Monitor GCD Convergence", ConceptType.NUMBER_THEORY,
        matched_primitive="monitor_gcd_convergence",
        inputs=[IOSpec(name="remainders", type_desc="ndarray")],
        outputs=[IOSpec(name="n_steps", type_desc="int"), IOSpec(name="avg_reduction_ratio", type_desc="float")],
        description="Monitor convergence rate of the Euclidean algorithm.",
        type_signature="ndarray -> tuple[int, float]",
    )
    rhs = CDGExport(nodes=[iterate, convergence, sink], edges=[_edge("iterate", "convergence"), _edge("convergence", "sink")])

    return RewriteRule(
        name="insert_gcd_convergence_monitoring_after_iterate", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"iterate": "iterate", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"iterate": "iterate", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_small_prime_check() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    conclude = _node("conclude", _CONCLUDE, ConceptType.NUMBER_THEORY)
    lhs = CDGExport(nodes=[src, conclude], edges=[_edge("src", "conclude")])
    interface = CDGExport(nodes=[src, conclude], edges=[])

    prime_check = _node(
        "prime_check", "Check Small Prime Divisors", ConceptType.NUMBER_THEORY,
        matched_primitive="check_small_prime_divisors",
        inputs=[IOSpec(name="n", type_desc="int"), IOSpec(name="n_primes", type_desc="int")],
        outputs=[IOSpec(name="has_small_factor", type_desc="bool"), IOSpec(name="smallest_factor", type_desc="int")],
        description="Check divisibility by small primes as a quick compositeness test.",
        type_signature="int, int -> tuple[bool, int]",
    )
    rhs = CDGExport(nodes=[src, prime_check, conclude], edges=[_edge("src", "prime_check"), _edge("prime_check", "conclude")])

    return RewriteRule(
        name="insert_small_prime_check_before_conclude", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "conclude": "conclude"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "conclude": "conclude"}, edge_map={}),
        priority=1,
    )


def _build_insert_modular_overflow_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    reduce = _node("reduce", _REDUCE, ConceptType.NUMBER_THEORY)
    lhs = CDGExport(nodes=[src, reduce], edges=[_edge("src", "reduce")])
    interface = CDGExport(nodes=[src, reduce], edges=[])

    overflow = _node(
        "overflow", "Detect Modular Overflow", ConceptType.NUMBER_THEORY,
        matched_primitive="detect_modular_overflow",
        inputs=[IOSpec(name="base", type_desc="int"), IOSpec(name="exponent", type_desc="int"), IOSpec(name="modulus", type_desc="int")],
        outputs=[IOSpec(name="would_overflow", type_desc="bool"), IOSpec(name="safe_bits_needed", type_desc="int")],
        description="Detect whether modular exponentiation risks intermediate overflow.",
        type_signature="int, int, int -> tuple[bool, int]",
    )
    rhs = CDGExport(nodes=[src, overflow, reduce], edges=[_edge("src", "overflow"), _edge("overflow", "reduce")])

    return RewriteRule(
        name="insert_modular_overflow_detection_before_reduce", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "reduce": "reduce"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "reduce": "reduce"}, edge_map={}),
        priority=2,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_input_range(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    n_risk = intermediates.get("input_overflow_risk_count")
    if n_risk is None:
        return None
    try:
        r = int(n_risk)
    except (ValueError, TypeError):
        return None
    if r > 0:
        return ExpansionDiagnostic(
            rule_name="insert_input_range_validation_before_reduce",
            severity=min(1.0, r / 5.0), evidence=f"{r} input value(s) risk overflow in intermediate computation",
            metric_name="input_overflow_risk_count", metric_value=float(r), threshold=0.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_gcd_convergence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    avg_ratio = intermediates.get("gcd_reduction_ratio")
    if avg_ratio is None:
        return None
    try:
        ratio = float(avg_ratio)
    except (ValueError, TypeError):
        return None
    if ratio > 0.618:  # Fibonacci worst case
        return ExpansionDiagnostic(
            rule_name="insert_gcd_convergence_monitoring_after_iterate",
            severity=min(1.0, (ratio - 0.618) / 0.382),
            evidence=f"GCD reduction ratio {ratio:.3f} exceeds 0.618 — slow convergence (Fibonacci-like input)",
            metric_name="gcd_reduction_ratio", metric_value=ratio, threshold=0.618, source_domain=_DOMAIN,
        )
    return None


def _diagnose_composite_without_sieve(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    has_small_factor = intermediates.get("has_small_factor")
    if has_small_factor is None:
        return None
    try:
        hsf = bool(has_small_factor)
    except (ValueError, TypeError):
        return None
    if hsf:
        return ExpansionDiagnostic(
            rule_name="insert_small_prime_check_before_conclude",
            severity=0.5,
            evidence="Number has small prime factor — expensive primality test can be skipped",
            metric_name="has_small_factor", metric_value=1.0, threshold=0.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_modular_overflow(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    would_overflow = intermediates.get("modular_overflow")
    if would_overflow is None:
        return None
    try:
        wo = bool(would_overflow)
    except (ValueError, TypeError):
        return None
    if wo:
        return ExpansionDiagnostic(
            rule_name="insert_modular_overflow_detection_before_reduce",
            severity=1.0,
            evidence="Modular exponentiation intermediate product exceeds int64 — needs arbitrary precision",
            metric_name="modular_overflow", metric_value=1.0, threshold=0.0, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class NumberTheoryExpansionRuleSet:
    """Expansion rules for number theory pipelines (GCD, Prime Sieve, Modular Exponentiation)."""

    name = "number_theory"
    domain = "number_theory"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_input_range_validation(),
            _build_insert_gcd_convergence_monitoring(),
            _build_insert_small_prime_check(),
            _build_insert_modular_overflow_detection(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_input_range, _diagnose_gcd_convergence, _diagnose_composite_without_sieve, _diagnose_modular_overflow]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

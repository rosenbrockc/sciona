"""Expansion rules for the Linear Algebra family (LU, QR, Cholesky, SVD, Eigendecomposition).

Linear Algebra skeleton topology (3 nodes, linear pipeline):

    Factorize → Solve/Transform → Validate

Expansion insertion points:
  - Before Factorize: conditioning check
  - After Factorize: decomposition accuracy validation
  - Before Solve/Transform: rank deficiency detection
  - After Solve/Transform: iterative convergence monitoring
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

_DOMAIN = "linear_algebra"

_FACTORIZE = "Factorize"
_SOLVE_TRANSFORM = "Solve/Transform"
_VALIDATE = "Validate"


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


def _build_insert_conditioning_check() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    factorize = _node("factorize", _FACTORIZE, ConceptType.ALGEBRA)
    lhs = CDGExport(nodes=[src, factorize], edges=[_edge("src", "factorize")])
    interface = CDGExport(nodes=[src, factorize], edges=[])

    conditioning = _node(
        "conditioning", "Check Matrix Conditioning", ConceptType.ALGEBRA,
        matched_primitive="check_matrix_conditioning",
        inputs=[IOSpec(name="A", type_desc="ndarray")],
        outputs=[IOSpec(name="condition_number", type_desc="float"),
                 IOSpec(name="is_well_conditioned", type_desc="bool")],
        description="Analyze the condition number of the input matrix.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, conditioning, factorize],
        edges=[_edge("src", "conditioning"), _edge("conditioning", "factorize")],
    )

    return RewriteRule(
        name="insert_conditioning_check_before_factorize",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "factorize": "factorize"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "factorize": "factorize"}, edge_map={}),
        priority=3,
    )


def _build_insert_decomposition_accuracy() -> RewriteRule:
    factorize = _node("factorize", _FACTORIZE, ConceptType.ALGEBRA)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[factorize, sink], edges=[_edge("factorize", "sink")])
    interface = CDGExport(nodes=[factorize, sink], edges=[])

    accuracy = _node(
        "accuracy", "Validate Decomposition Accuracy", ConceptType.ALGEBRA,
        matched_primitive="validate_decomposition_accuracy",
        inputs=[IOSpec(name="A", type_desc="ndarray"),
                IOSpec(name="reconstructed", type_desc="ndarray")],
        outputs=[IOSpec(name="relative_error", type_desc="float"),
                 IOSpec(name="is_accurate", type_desc="bool")],
        description="Check residual ||A - reconstructed|| / ||A||.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[factorize, accuracy, sink],
        edges=[_edge("factorize", "accuracy"), _edge("accuracy", "sink")],
    )

    return RewriteRule(
        name="insert_decomposition_accuracy_after_factorize",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"factorize": "factorize", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"factorize": "factorize", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_rank_deficiency_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    solve = _node("solve", _SOLVE_TRANSFORM, ConceptType.ALGEBRA)
    lhs = CDGExport(nodes=[src, solve], edges=[_edge("src", "solve")])
    interface = CDGExport(nodes=[src, solve], edges=[])

    rank = _node(
        "rank", "Detect Rank Deficiency", ConceptType.ALGEBRA,
        matched_primitive="detect_rank_deficiency",
        inputs=[IOSpec(name="singular_values", type_desc="ndarray"),
                IOSpec(name="expected_rank", type_desc="int")],
        outputs=[IOSpec(name="effective_rank", type_desc="int"),
                 IOSpec(name="is_full_rank", type_desc="bool")],
        description="Estimate numerical rank vs expected rank.",
        type_signature="ndarray, int -> tuple[int, bool]",
    )
    rhs = CDGExport(
        nodes=[src, rank, solve],
        edges=[_edge("src", "rank"), _edge("rank", "solve")],
    )

    return RewriteRule(
        name="insert_rank_deficiency_detection_before_solve",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "solve": "solve"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "solve": "solve"}, edge_map={}),
        priority=2,
    )


def _build_insert_iterative_convergence() -> RewriteRule:
    solve = _node("solve", _SOLVE_TRANSFORM, ConceptType.ALGEBRA)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[solve, sink], edges=[_edge("solve", "sink")])
    interface = CDGExport(nodes=[solve, sink], edges=[])

    convergence = _node(
        "convergence", "Monitor Iterative Convergence", ConceptType.ALGEBRA,
        matched_primitive="monitor_iterative_convergence",
        inputs=[IOSpec(name="residual_norms", type_desc="ndarray")],
        outputs=[IOSpec(name="convergence_rate", type_desc="float"),
                 IOSpec(name="is_converging", type_desc="bool")],
        description="Track residual norm decay for iterative solvers.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[solve, convergence, sink],
        edges=[_edge("solve", "convergence"), _edge("convergence", "sink")],
    )

    return RewriteRule(
        name="insert_iterative_convergence_after_solve",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"solve": "solve", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"solve": "solve", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_conditioning(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    cond = intermediates.get("matrix_condition_number")
    if cond is None:
        return None
    try:
        c = float(cond)
    except (ValueError, TypeError):
        return None
    if c > 1e12:
        return ExpansionDiagnostic(
            rule_name="insert_conditioning_check_before_factorize",
            severity=min(1.0, np.log10(max(c, 1)) / 15.0),
            evidence=f"Matrix condition number {c:.2e} exceeds 1e12 — ill-conditioned",
            metric_name="matrix_condition_number", metric_value=c, threshold=1e12,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_decomposition_accuracy(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    residual = intermediates.get("decomposition_residual")
    if residual is None:
        return None
    try:
        r = float(residual)
    except (ValueError, TypeError):
        return None
    if r > 1e-8:
        return ExpansionDiagnostic(
            rule_name="insert_decomposition_accuracy_after_factorize",
            severity=min(1.0, np.log10(max(r, 1e-30)) / -8.0 + 1.0),
            evidence=f"Decomposition residual {r:.2e} exceeds 1e-8 — inaccurate factorization",
            metric_name="decomposition_residual", metric_value=r, threshold=1e-8,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_rank_deficiency(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    deficit = intermediates.get("rank_deficit")
    if deficit is None:
        return None
    try:
        d = int(deficit)
    except (ValueError, TypeError):
        return None
    if d > 0:
        return ExpansionDiagnostic(
            rule_name="insert_rank_deficiency_detection_before_solve",
            severity=min(1.0, d / 5.0),
            evidence=f"Rank deficit of {d} — matrix is rank-deficient",
            metric_name="rank_deficit", metric_value=float(d), threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_iterative_convergence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    rate = intermediates.get("iterative_convergence_rate")
    if rate is None:
        return None
    try:
        r = float(rate)
    except (ValueError, TypeError):
        return None
    if r > 0.99:
        return ExpansionDiagnostic(
            rule_name="insert_iterative_convergence_after_solve",
            severity=min(1.0, (r - 0.99) / 0.01),
            evidence=f"Iterative convergence rate {r:.4f} exceeds 0.99 — stalling",
            metric_name="iterative_convergence_rate", metric_value=r, threshold=0.99,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class LinearAlgebraExpansionRuleSet:
    """Expansion rules for linear algebra pipelines (LU, QR, Cholesky, SVD, Eigendecomposition)."""

    name = "linear_algebra"
    domain = "linear_algebra"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_conditioning_check(),
            _build_insert_decomposition_accuracy(),
            _build_insert_rank_deficiency_detection(),
            _build_insert_iterative_convergence(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_conditioning, _diagnose_decomposition_accuracy,
                    _diagnose_rank_deficiency, _diagnose_iterative_convergence]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

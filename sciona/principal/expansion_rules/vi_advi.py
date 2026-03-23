"""Expansion rules for the VI/ADVI family (ADVI, Mean-field VI, Full-rank VI).

VI/ADVI skeleton topology (4 nodes, with fan-out):

    Shape Alloc → Reparameterization → ELBO Eval → L-BFGS Optimizer

Expansion insertion points:
  - After ELBO Eval: ELBO convergence monitoring
  - After Reparameterization: gradient variance analysis
  - After Shape Alloc: posterior collapse detection
  - After L-BFGS Optimizer: step size stability check
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

_DOMAIN = "vi_advi"

_SHAPE_ALLOC = "Shape Alloc"
_REPARAMETERIZATION = "Reparameterization"
_ELBO_EVAL = "ELBO Eval"
_LBFGS_OPTIMIZER = "L-BFGS Optimizer"


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


def _build_insert_elbo_convergence_monitoring() -> RewriteRule:
    elbo = _node("elbo", _ELBO_EVAL, ConceptType.PROBABILISTIC_ORACLE)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[elbo, sink], edges=[_edge("elbo", "sink")])
    interface = CDGExport(nodes=[elbo, sink], edges=[])

    convergence = _node(
        "convergence", "Monitor ELBO Convergence", ConceptType.VI_ELBO,
        matched_primitive="monitor_elbo_convergence",
        inputs=[IOSpec(name="elbo_history", type_desc="ndarray"), IOSpec(name="window", type_desc="int")],
        outputs=[IOSpec(name="relative_improvement", type_desc="float"), IOSpec(name="has_converged", type_desc="bool")],
        description="Monitor ELBO convergence from optimization history.",
        type_signature="ndarray, int -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[elbo, convergence, sink], edges=[_edge("elbo", "convergence"), _edge("convergence", "sink")])

    return RewriteRule(
        name="insert_elbo_convergence_monitoring_after_elbo_eval", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"elbo": "elbo", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"elbo": "elbo", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_gradient_variance_analysis() -> RewriteRule:
    reparam = _node("reparam", _REPARAMETERIZATION, ConceptType.VI_ELBO)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[reparam, sink], edges=[_edge("reparam", "sink")])
    interface = CDGExport(nodes=[reparam, sink], edges=[])

    grad_var = _node(
        "grad_var", "Analyze Gradient Variance", ConceptType.VI_ELBO,
        matched_primitive="analyze_gradient_variance",
        inputs=[IOSpec(name="gradient_samples", type_desc="ndarray")],
        outputs=[IOSpec(name="mean_cv", type_desc="float"), IOSpec(name="is_low_variance", type_desc="bool")],
        description="Analyze variance of stochastic gradient estimates.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[reparam, grad_var, sink], edges=[_edge("reparam", "grad_var"), _edge("grad_var", "sink")])

    return RewriteRule(
        name="insert_gradient_variance_analysis_after_reparameterization", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"reparam": "reparam", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"reparam": "reparam", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_posterior_collapse_detection() -> RewriteRule:
    shape = _node("shape", _SHAPE_ALLOC, ConceptType.VI_ELBO)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[shape, sink], edges=[_edge("shape", "sink")])
    interface = CDGExport(nodes=[shape, sink], edges=[])

    collapse = _node(
        "collapse", "Detect Posterior Collapse", ConceptType.VI_ELBO,
        matched_primitive="detect_posterior_collapse",
        inputs=[IOSpec(name="kl_per_dimension", type_desc="ndarray"), IOSpec(name="threshold", type_desc="float")],
        outputs=[IOSpec(name="n_collapsed", type_desc="int"), IOSpec(name="collapse_fraction", type_desc="float")],
        description="Detect posterior collapse (KL vanishing) per latent dimension.",
        type_signature="ndarray, float -> tuple[int, float]",
    )
    rhs = CDGExport(nodes=[shape, collapse, sink], edges=[_edge("shape", "collapse"), _edge("collapse", "sink")])

    return RewriteRule(
        name="insert_posterior_collapse_detection_after_shape_alloc", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"shape": "shape", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"shape": "shape", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_step_size_stability_check() -> RewriteRule:
    optimizer = _node("optimizer", _LBFGS_OPTIMIZER, ConceptType.VI_ELBO)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[optimizer, sink], edges=[_edge("optimizer", "sink")])
    interface = CDGExport(nodes=[optimizer, sink], edges=[])

    step_check = _node(
        "step_check", "Check Step Size Stability", ConceptType.VI_ELBO,
        matched_primitive="check_step_size_stability",
        inputs=[IOSpec(name="step_sizes", type_desc="ndarray")],
        outputs=[IOSpec(name="coefficient_of_variation", type_desc="float"), IOSpec(name="is_stable", type_desc="bool")],
        description="Check whether optimizer step sizes are stable.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[optimizer, step_check, sink], edges=[_edge("optimizer", "step_check"), _edge("step_check", "sink")])

    return RewriteRule(
        name="insert_step_size_stability_check_after_optimizer", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"optimizer": "optimizer", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"optimizer": "optimizer", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_elbo_convergence(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    rel = intermediates.get("elbo_relative_improvement")
    if rel is None:
        return None
    try:
        r = float(rel)
    except (ValueError, TypeError):
        return None
    if r > 0.01:
        return ExpansionDiagnostic(
            rule_name="insert_elbo_convergence_monitoring_after_elbo_eval",
            severity=min(1.0, r * 10), evidence=f"ELBO relative improvement {r:.4f} exceeds 0.01 — not converged",
            metric_name="elbo_relative_improvement", metric_value=r, threshold=0.01, source_domain=_DOMAIN,
        )
    return None


def _diagnose_gradient_variance(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    cv = intermediates.get("gradient_mean_cv")
    if cv is None:
        return None
    try:
        c = float(cv)
    except (ValueError, TypeError):
        return None
    if c > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_gradient_variance_analysis_after_reparameterization",
            severity=min(1.0, c / 5.0), evidence=f"Gradient CV {c:.2f} exceeds 1.0 — noisy gradients",
            metric_name="gradient_mean_cv", metric_value=c, threshold=1.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_posterior_collapse(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("posterior_collapse_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_posterior_collapse_detection_after_shape_alloc",
            severity=min(1.0, f * 2), evidence=f"Posterior collapse fraction {f:.2f} exceeds 0.1 — {f*100:.0f}% dims collapsed",
            metric_name="posterior_collapse_fraction", metric_value=f, threshold=0.1, source_domain=_DOMAIN,
        )
    return None


def _diagnose_step_size_stability(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    cv = intermediates.get("step_size_cv")
    if cv is None:
        return None
    try:
        c = float(cv)
    except (ValueError, TypeError):
        return None
    if c > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_step_size_stability_check_after_optimizer",
            severity=min(1.0, c), evidence=f"Step size CV {c:.2f} exceeds 0.5 — unstable optimization",
            metric_name="step_size_cv", metric_value=c, threshold=0.5, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class VIADVIExpansionRuleSet:
    """Expansion rules for VI/ADVI pipelines (ADVI, Mean-field VI, Full-rank VI)."""

    name = "vi_advi"
    domain = "vi_advi"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_elbo_convergence_monitoring(),
            _build_insert_gradient_variance_analysis(),
            _build_insert_posterior_collapse_detection(),
            _build_insert_step_size_stability_check(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_elbo_convergence, _diagnose_gradient_variance, _diagnose_posterior_collapse, _diagnose_step_size_stability]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

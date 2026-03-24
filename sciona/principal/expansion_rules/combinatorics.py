"""Expansion rules for the Combinatorial Optimization family (Branch & Bound, CSP, SAT).

Combinatorial Optimization skeleton topology (4 nodes, linear pipeline):

    Bound → Branch → Prune → Select

Expansion insertion points:
  - After Branch: branching factor analysis
  - After Bound: bound tightness monitoring
  - Before Branch: symmetry detection
  - After Prune: pruning effectiveness check
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

_DOMAIN = "combinatorics"

_BOUND = "Bound"
_BRANCH = "Branch"
_PRUNE = "Prune"
_SELECT = "Select"


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


def _build_insert_branching_factor_analysis() -> RewriteRule:
    branch = _node("branch", _BRANCH, ConceptType.COMBINATORICS)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[branch, sink], edges=[_edge("branch", "sink")])
    interface = CDGExport(nodes=[branch, sink], edges=[])

    branching = _node(
        "branching", "Analyze Branching Factor", ConceptType.COMBINATORICS,
        matched_primitive="analyze_branching_factor",
        inputs=[IOSpec(name="child_counts", type_desc="ndarray")],
        outputs=[IOSpec(name="mean_branching", type_desc="float"),
                 IOSpec(name="is_manageable", type_desc="bool")],
        description="Analyze effective branching factor of the search tree.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[branch, branching, sink],
        edges=[_edge("branch", "branching"), _edge("branching", "sink")],
    )

    return RewriteRule(
        name="insert_branching_factor_analysis_after_branch",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"branch": "branch", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"branch": "branch", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_bound_tightness_monitoring() -> RewriteRule:
    bound = _node("bound", _BOUND, ConceptType.COMBINATORICS)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[bound, sink], edges=[_edge("bound", "sink")])
    interface = CDGExport(nodes=[bound, sink], edges=[])

    tightness = _node(
        "tightness", "Monitor Bound Tightness", ConceptType.COMBINATORICS,
        matched_primitive="monitor_bound_tightness",
        inputs=[IOSpec(name="upper_bounds", type_desc="ndarray"),
                IOSpec(name="lower_bounds", type_desc="ndarray")],
        outputs=[IOSpec(name="gap_ratio", type_desc="float"),
                 IOSpec(name="is_tight", type_desc="bool")],
        description="Track gap between upper and lower bounds.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[bound, tightness, sink],
        edges=[_edge("bound", "tightness"), _edge("tightness", "sink")],
    )

    return RewriteRule(
        name="insert_bound_tightness_monitoring_after_bound",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"bound": "bound", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"bound": "bound", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_symmetry_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    branch = _node("branch", _BRANCH, ConceptType.COMBINATORICS)
    lhs = CDGExport(nodes=[src, branch], edges=[_edge("src", "branch")])
    interface = CDGExport(nodes=[src, branch], edges=[])

    symmetry = _node(
        "symmetry", "Detect Symmetry", ConceptType.COMBINATORICS,
        matched_primitive="detect_symmetry",
        inputs=[IOSpec(name="candidate_pairs", type_desc="ndarray"),
                IOSpec(name="equivalence_count", type_desc="int")],
        outputs=[IOSpec(name="symmetry_fraction", type_desc="float"),
                 IOSpec(name="has_symmetry", type_desc="bool")],
        description="Identify symmetry in the search space for breaking.",
        type_signature="ndarray, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, symmetry, branch],
        edges=[_edge("src", "symmetry"), _edge("symmetry", "branch")],
    )

    return RewriteRule(
        name="insert_symmetry_detection_before_branch",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "branch": "branch"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "branch": "branch"}, edge_map={}),
        priority=2,
    )


def _build_insert_pruning_effectiveness() -> RewriteRule:
    prune = _node("prune", _PRUNE, ConceptType.COMBINATORICS)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[prune, sink], edges=[_edge("prune", "sink")])
    interface = CDGExport(nodes=[prune, sink], edges=[])

    pruning = _node(
        "pruning", "Check Pruning Effectiveness", ConceptType.COMBINATORICS,
        matched_primitive="check_pruning_effectiveness",
        inputs=[IOSpec(name="total_nodes", type_desc="int"),
                IOSpec(name="pruned_nodes", type_desc="int")],
        outputs=[IOSpec(name="pruning_rate", type_desc="float"),
                 IOSpec(name="is_effective", type_desc="bool")],
        description="Check fraction of subtrees pruned vs explored.",
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[prune, pruning, sink],
        edges=[_edge("prune", "pruning"), _edge("pruning", "sink")],
    )

    return RewriteRule(
        name="insert_pruning_effectiveness_after_prune",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"prune": "prune", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"prune": "prune", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_branching_factor(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    bf = intermediates.get("mean_branching_factor")
    if bf is None:
        return None
    try:
        b = float(bf)
    except (ValueError, TypeError):
        return None
    if b > 10:
        return ExpansionDiagnostic(
            rule_name="insert_branching_factor_analysis_after_branch",
            severity=min(1.0, (b - 10) / 20.0),
            evidence=f"Mean branching factor {b:.1f} exceeds 10 — exponential blowup risk",
            metric_name="mean_branching_factor", metric_value=b, threshold=10.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_bound_tightness(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    gap = intermediates.get("bound_gap_ratio")
    if gap is None:
        return None
    try:
        g = float(gap)
    except (ValueError, TypeError):
        return None
    if g > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_bound_tightness_monitoring_after_bound",
            severity=min(1.0, (g - 0.5) / 0.5),
            evidence=f"Bound gap ratio {g:.2f} exceeds 0.5 — loose bounds",
            metric_name="bound_gap_ratio", metric_value=g, threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_symmetry(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    sf = intermediates.get("symmetry_fraction")
    if sf is None:
        return None
    try:
        s = float(sf)
    except (ValueError, TypeError):
        return None
    if s > 0.3:
        return ExpansionDiagnostic(
            rule_name="insert_symmetry_detection_before_branch",
            severity=min(1.0, (s - 0.3) / 0.7),
            evidence=f"Symmetry fraction {s:.2f} exceeds 0.3 — add symmetry-breaking constraints",
            metric_name="symmetry_fraction", metric_value=s, threshold=0.3,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_pruning(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    pr = intermediates.get("pruning_rate")
    if pr is None:
        return None
    try:
        p = float(pr)
    except (ValueError, TypeError):
        return None
    if p < 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_pruning_effectiveness_after_prune",
            severity=min(1.0, (0.1 - p) / 0.1),
            evidence=f"Pruning rate {p:.3f} is below 0.1 — ineffective pruning",
            metric_name="pruning_rate", metric_value=p, threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class CombinatoricsExpansionRuleSet:
    """Expansion rules for combinatorial optimization pipelines (B&B, CSP, SAT)."""

    name = "combinatorics"
    domain = "combinatorics"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_branching_factor_analysis(),
            _build_insert_bound_tightness_monitoring(),
            _build_insert_symmetry_detection(),
            _build_insert_pruning_effectiveness(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_branching_factor, _diagnose_bound_tightness,
                    _diagnose_symmetry, _diagnose_pruning]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

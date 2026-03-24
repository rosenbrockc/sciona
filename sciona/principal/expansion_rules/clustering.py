"""Expansion rules for the Clustering family (K-Means, K-Medoids, EM/GMM).

Clustering skeleton topology (3 nodes, iterative):

    Initialize Centers -> Assign Points -> Update Centers

Expansion insertion points:
  - After Assign Points: cluster balance analysis
  - After Update Centers: assignment stability monitoring
  - Before Assign Points: empty cluster detection
  - Before Update Centers: separation validation
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

_DOMAIN = "clustering"

_INITIALIZE_CENTERS = "Initialize Centers"
_ASSIGN_POINTS = "Assign Points"
_UPDATE_CENTERS = "Update Centers"


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


def _build_insert_cluster_balance_analysis() -> RewriteRule:
    assign = _node("assign", _ASSIGN_POINTS, ConceptType.CLUSTERING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[assign, sink], edges=[_edge("assign", "sink")])
    interface = CDGExport(nodes=[assign, sink], edges=[])

    balance = _node(
        "balance", "Analyze Cluster Balance", ConceptType.CLUSTERING,
        matched_primitive="analyze_cluster_balance",
        inputs=[IOSpec(name="cluster_sizes", type_desc="ndarray")],
        outputs=[IOSpec(name="imbalance_ratio", type_desc="float"),
                 IOSpec(name="is_balanced", type_desc="bool")],
        description="Check max/min cluster size ratio.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[assign, balance, sink],
        edges=[_edge("assign", "balance"), _edge("balance", "sink")],
    )

    return RewriteRule(
        name="insert_cluster_balance_analysis_after_assign",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"assign": "assign", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"assign": "assign", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_assignment_stability() -> RewriteRule:
    update = _node("update", _UPDATE_CENTERS, ConceptType.CLUSTERING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    stability = _node(
        "stability", "Monitor Assignment Stability", ConceptType.CLUSTERING,
        matched_primitive="monitor_assignment_stability",
        inputs=[IOSpec(name="prev_assignments", type_desc="ndarray"),
                IOSpec(name="curr_assignments", type_desc="ndarray")],
        outputs=[IOSpec(name="change_fraction", type_desc="float"),
                 IOSpec(name="is_stable", type_desc="bool")],
        description="Track fraction of points changing cluster assignment.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[update, stability, sink],
        edges=[_edge("update", "stability"), _edge("stability", "sink")],
    )

    return RewriteRule(
        name="insert_assignment_stability_after_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_empty_cluster_detection() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    assign = _node("assign", _ASSIGN_POINTS, ConceptType.CLUSTERING)
    lhs = CDGExport(nodes=[src, assign], edges=[_edge("src", "assign")])
    interface = CDGExport(nodes=[src, assign], edges=[])

    empty = _node(
        "empty", "Detect Empty Clusters", ConceptType.CLUSTERING,
        matched_primitive="detect_empty_clusters",
        inputs=[IOSpec(name="cluster_sizes", type_desc="ndarray")],
        outputs=[IOSpec(name="n_empty", type_desc="int"),
                 IOSpec(name="has_empty", type_desc="bool")],
        description="Count clusters with zero members.",
        type_signature="ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(
        nodes=[src, empty, assign],
        edges=[_edge("src", "empty"), _edge("empty", "assign")],
    )

    return RewriteRule(
        name="insert_empty_cluster_detection_after_assign",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "assign": "assign"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "assign": "assign"}, edge_map={}),
        priority=2,
    )


def _build_insert_separation_validation() -> RewriteRule:
    src = _node("src", "source", ConceptType.CUSTOM)
    update = _node("update", _UPDATE_CENTERS, ConceptType.CLUSTERING)
    lhs = CDGExport(nodes=[src, update], edges=[_edge("src", "update")])
    interface = CDGExport(nodes=[src, update], edges=[])

    separation = _node(
        "separation", "Validate Separation", ConceptType.CLUSTERING,
        matched_primitive="validate_separation",
        inputs=[IOSpec(name="inter_distances", type_desc="ndarray"),
                IOSpec(name="intra_distances", type_desc="ndarray")],
        outputs=[IOSpec(name="separation_ratio", type_desc="float"),
                 IOSpec(name="is_well_separated", type_desc="bool")],
        description="Check inter/intra cluster distance ratio.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, separation, update],
        edges=[_edge("src", "separation"), _edge("separation", "update")],
    )

    return RewriteRule(
        name="insert_separation_validation_after_update",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "update": "update"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "update": "update"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_cluster_balance(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ratio = intermediates.get("cluster_imbalance_ratio")
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return None
    if r > 10.0:
        return ExpansionDiagnostic(
            rule_name="insert_cluster_balance_analysis_after_assign",
            severity=min(1.0, np.log10(max(r, 1)) / 3.0),
            evidence=f"Cluster imbalance ratio {r:.2f} exceeds 10.0 — highly imbalanced",
            metric_name="cluster_imbalance_ratio", metric_value=r, threshold=10.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_assignment_stability(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("assignment_change_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.01:
        return ExpansionDiagnostic(
            rule_name="insert_assignment_stability_after_update",
            severity=min(1.0, f),
            evidence=f"Assignment change fraction {f:.4f} exceeds 0.01 — unstable clustering",
            metric_name="assignment_change_fraction", metric_value=f, threshold=0.01,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_empty_clusters(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    n_empty = intermediates.get("n_empty_clusters")
    if n_empty is None:
        return None
    try:
        n = int(n_empty)
    except (ValueError, TypeError):
        return None
    if n > 0:
        return ExpansionDiagnostic(
            rule_name="insert_empty_cluster_detection_after_assign",
            severity=min(1.0, n / 5.0),
            evidence=f"{n} empty cluster(s) detected — degenerate solution",
            metric_name="n_empty_clusters", metric_value=float(n), threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_separation(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ratio = intermediates.get("separation_ratio")
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return None
    if r < 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_separation_validation_after_update",
            severity=min(1.0, 1.0 - r),
            evidence=f"Separation ratio {r:.4f} below 1.0 — poorly separated clusters",
            metric_name="separation_ratio", metric_value=r, threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class ClusteringExpansionRuleSet:
    """Expansion rules for clustering pipelines (K-Means, K-Medoids, EM/GMM)."""

    name = "clustering"
    domain = "clustering"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_cluster_balance_analysis(),
            _build_insert_assignment_stability(),
            _build_insert_empty_cluster_detection(),
            _build_insert_separation_validation(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_cluster_balance, _diagnose_assignment_stability,
                    _diagnose_empty_clusters, _diagnose_separation]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

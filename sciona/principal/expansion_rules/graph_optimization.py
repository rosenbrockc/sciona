"""Expansion rules for the Graph Optimization family (Dijkstra, Bellman-Ford, Floyd-Warshall).

Defines DPO rules and diagnostic functions that let the expansion engine
insert negative weight detection, relaxation convergence monitoring,
distance overflow detection, and graph density analysis into graph
optimization CDGs.

Graph optimization skeleton topology (4 nodes, linear pipeline):

    Init Weights → Relax Edges → Check Negative Cycle → Extract Path

Expansion insertion points:
  - Before Relax Edges: negative weight detection, graph density analysis
  - After Relax Edges: relaxation convergence monitoring
  - Before Extract Path: distance overflow detection

All diagnostics are pure functions of graph optimization intermediates.
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

_DOMAIN = "graph_optimization"

# Graph optimization skeleton node names
_INIT_WEIGHTS = "Init Weights"
_RELAX_EDGES = "Relax Edges"
_CHECK_NEGATIVE_CYCLE = "Check Negative Cycle"
_EXTRACT_PATH = "Extract Path"


# ---------------------------------------------------------------------------
# Node / edge helpers
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_negative_weight_detection() -> RewriteRule:
    """Interpose ``detect_negative_weights`` before Relax Edges.

    Dijkstra's algorithm silently produces incorrect results with
    negative edge weights.  This pre-check flags negative weights
    so the user can switch to Bellman-Ford.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    relax = _node(
        "relax",
        _RELAX_EDGES,
        ConceptType.GRAPH_OPTIMIZATION,
    )
    lhs = CDGExport(nodes=[src, relax], edges=[_edge("src", "relax")])
    interface = CDGExport(nodes=[src, relax], edges=[])

    neg_check = _node(
        "neg_check",
        "Detect Negative Weights",
        ConceptType.GRAPH_OPTIMIZATION,
        matched_primitive="detect_negative_weights",
        inputs=[
            IOSpec(name="edge_weights", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="n_negative", type_desc="int"),
            IOSpec(name="min_weight", type_desc="float"),
        ],
        description="Detect negative edge weights that invalidate Dijkstra's algorithm.",
        type_signature="ndarray -> tuple[int, float]",
    )
    rhs = CDGExport(
        nodes=[src, neg_check, relax],
        edges=[
            _edge("src", "neg_check"),
            _edge("neg_check", "relax"),
        ],
    )

    return RewriteRule(
        name="insert_negative_weight_detection_before_relax",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "relax": "relax"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "relax": "relax"}, edge_map={}),
        priority=3,
    )


def _build_insert_relaxation_convergence() -> RewriteRule:
    """Interpose ``monitor_relaxation_convergence`` after Relax Edges.

    When relaxation converges early, remaining iterations can be
    skipped for significant speedup (especially Bellman-Ford).
    """
    relax = _node(
        "relax",
        _RELAX_EDGES,
        ConceptType.GRAPH_OPTIMIZATION,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[relax, sink], edges=[_edge("relax", "sink")])
    interface = CDGExport(nodes=[relax, sink], edges=[])

    convergence = _node(
        "convergence",
        "Monitor Relaxation Convergence",
        ConceptType.GRAPH_OPTIMIZATION,
        matched_primitive="monitor_relaxation_convergence",
        inputs=[
            IOSpec(name="distance_snapshots", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="converged_at", type_desc="int"),
            IOSpec(name="has_converged", type_desc="bool"),
        ],
        description="Monitor whether edge relaxation has converged.",
        type_signature="ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(
        nodes=[relax, convergence, sink],
        edges=[
            _edge("relax", "convergence"),
            _edge("convergence", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_relaxation_convergence_after_relax",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"relax": "relax", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"relax": "relax", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_distance_overflow_detection() -> RewriteRule:
    """Interpose ``detect_distance_overflow`` before Extract Path.

    Large edge weights can cause numeric overflow during relaxation,
    producing incorrect shortest paths.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    extract = _node(
        "extract",
        _EXTRACT_PATH,
        ConceptType.GRAPH_OPTIMIZATION,
    )
    lhs = CDGExport(nodes=[src, extract], edges=[_edge("src", "extract")])
    interface = CDGExport(nodes=[src, extract], edges=[])

    overflow = _node(
        "overflow",
        "Detect Distance Overflow",
        ConceptType.GRAPH_OPTIMIZATION,
        matched_primitive="detect_distance_overflow",
        inputs=[
            IOSpec(name="distances", type_desc="ndarray"),
            IOSpec(name="overflow_threshold", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="n_overflow", type_desc="int"),
            IOSpec(name="max_distance", type_desc="float"),
        ],
        description="Detect numeric overflow in distance computations.",
        type_signature="ndarray, float -> tuple[int, float]",
    )
    rhs = CDGExport(
        nodes=[src, overflow, extract],
        edges=[
            _edge("src", "overflow"),
            _edge("overflow", "extract"),
        ],
    )

    return RewriteRule(
        name="insert_distance_overflow_detection_before_extract",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "extract": "extract"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "extract": "extract"}, edge_map={}),
        priority=2,
    )


def _build_insert_graph_density_analysis() -> RewriteRule:
    """Interpose ``analyze_graph_density`` before Relax Edges.

    Algorithm choice depends on graph density: sparse graphs favor
    Dijkstra with heap, dense graphs favor Floyd-Warshall.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    relax = _node(
        "relax",
        _RELAX_EDGES,
        ConceptType.GRAPH_OPTIMIZATION,
    )
    lhs = CDGExport(nodes=[src, relax], edges=[_edge("src", "relax")])
    interface = CDGExport(nodes=[src, relax], edges=[])

    density = _node(
        "density",
        "Analyze Graph Density",
        ConceptType.GRAPH_OPTIMIZATION,
        matched_primitive="analyze_graph_density",
        inputs=[
            IOSpec(name="n_nodes", type_desc="int"),
            IOSpec(name="n_edges", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="density", type_desc="float"),
            IOSpec(name="recommendation", type_desc="str"),
        ],
        description="Analyze graph density for algorithm selection guidance.",
        type_signature="int, int -> tuple[float, str]",
    )
    rhs = CDGExport(
        nodes=[src, density, relax],
        edges=[
            _edge("src", "density"),
            _edge("density", "relax"),
        ],
    )

    return RewriteRule(
        name="insert_graph_density_analysis_before_relax",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "relax": "relax"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "relax": "relax"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_negative_weights(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect negative edge weights."""
    intermediates = context.intermediates or {}
    min_weight = intermediates.get("min_edge_weight")

    if min_weight is None:
        return None

    try:
        w = float(min_weight)
    except (ValueError, TypeError):
        return None

    if w < 0:
        return ExpansionDiagnostic(
            rule_name="insert_negative_weight_detection_before_relax",
            severity=min(1.0, abs(w) / 100.0),
            evidence=(
                f"Minimum edge weight {w:.4f} is negative "
                f"— Dijkstra will produce incorrect results"
            ),
            metric_name="min_edge_weight",
            metric_value=w,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_relaxation_convergence(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect early relaxation convergence."""
    intermediates = context.intermediates or {}
    n_iterations = intermediates.get("relaxation_iterations")
    n_nodes = intermediates.get("n_nodes")

    if n_iterations is None or n_nodes is None:
        return None

    try:
        iters = int(n_iterations)
        n = int(n_nodes)
    except (ValueError, TypeError):
        return None

    if n <= 1:
        return None

    # Bellman-Ford needs at most n-1 iterations; if using far fewer, flag it
    utilization = iters / max(n - 1, 1)

    if utilization < 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_relaxation_convergence_after_relax",
            severity=min(1.0, (0.5 - utilization) / 0.5),
            evidence=(
                f"Relaxation used {iters}/{n - 1} iterations ({utilization:.0%}) "
                f"— early convergence detection can skip remaining passes"
            ),
            metric_name="relaxation_utilization",
            metric_value=utilization,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_distance_overflow(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect distance values approaching overflow."""
    intermediates = context.intermediates or {}
    max_distance = intermediates.get("max_distance")

    if max_distance is None:
        return None

    try:
        d = float(max_distance)
    except (ValueError, TypeError):
        return None

    threshold = 1e15
    if d > threshold:
        return ExpansionDiagnostic(
            rule_name="insert_distance_overflow_detection_before_extract",
            severity=min(1.0, np.log10(max(d / threshold, 1.0)) / 3.0),
            evidence=(
                f"Maximum distance {d:.2e} exceeds {threshold:.0e} threshold "
                f"— numeric overflow risk in path computation"
            ),
            metric_name="max_distance",
            metric_value=d,
            threshold=threshold,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_graph_density(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect dense graphs where Floyd-Warshall may be more efficient."""
    intermediates = context.intermediates or {}
    graph_density = intermediates.get("graph_density")

    if graph_density is None:
        return None

    try:
        density = float(graph_density)
    except (ValueError, TypeError):
        return None

    if density > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_graph_density_analysis_before_relax",
            severity=min(1.0, (density - 0.5) / 0.5),
            evidence=(
                f"Graph density {density:.2f} exceeds 0.5 threshold "
                f"— consider Floyd-Warshall over Dijkstra/Bellman-Ford"
            ),
            metric_name="graph_density",
            metric_value=density,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class GraphOptimizationExpansionRuleSet:
    """Expansion rules for graph optimization pipelines (Dijkstra, Bellman-Ford, Floyd-Warshall)."""

    name = "graph_optimization"
    domain = "graph_optimization"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_negative_weight_detection(),
            _build_insert_relaxation_convergence(),
            _build_insert_distance_overflow_detection(),
            _build_insert_graph_density_analysis(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        neg = _diagnose_negative_weights(cdg, context)
        if neg is not None:
            diagnostics.append(neg)

        conv = _diagnose_relaxation_convergence(cdg, context)
        if conv is not None:
            diagnostics.append(conv)

        overflow = _diagnose_distance_overflow(cdg, context)
        if overflow is not None:
            diagnostics.append(overflow)

        density = _diagnose_graph_density(cdg, context)
        if density is not None:
            diagnostics.append(density)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

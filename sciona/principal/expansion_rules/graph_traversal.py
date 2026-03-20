"""Expansion rules for the Graph Traversal family (BFS / DFS / Topological Sort).

Defines DPO rules and diagnostic functions that let the expansion engine
insert cycle detection, connectivity checks, frontier overflow detection,
and visited-set compaction into graph traversal CDGs.

Graph traversal skeleton topology (5 nodes, linear pipeline):

    Init Visited → Pick Next → Process Node → Update Frontier → Check Termination

Expansion insertion points:
  - Before Init Visited: cycle detection, connectivity check
  - After Update Frontier: frontier overflow detection, visited-set compaction

All diagnostics are pure functions of graph structure and traversal intermediates.
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

_DOMAIN = "graph_traversal"

# DFS color states (mirrored from runtime_graph_traversal)
_WHITE, _GRAY, _BLACK = 0, 1, 2

# Graph traversal skeleton node names
_INIT_VISITED = "Init Visited"
_PICK_NEXT = "Pick Next"
_PROCESS_NODE = "Process Node"
_UPDATE_FRONTIER = "Update Frontier"
_CHECK_TERMINATION = "Check Termination"


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


def _build_insert_cycle_detection() -> RewriteRule:
    """Interpose ``detect_cycles`` before Init Visited.

    For DAG-only algorithms (topological sort), undetected cycles cause
    infinite loops or incorrect results.  This pre-check aborts early
    with a clear diagnostic.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    init = _node(
        "init",
        _INIT_VISITED,
        ConceptType.GRAPH_TRAVERSAL,
    )
    lhs = CDGExport(nodes=[src, init], edges=[_edge("src", "init")])
    interface = CDGExport(nodes=[src, init], edges=[])

    cycle_check = _node(
        "cycle_check",
        "Detect Cycles",
        ConceptType.GRAPH_TRAVERSAL,
        matched_primitive="detect_cycles",
        inputs=[
            IOSpec(name="adjacency", type_desc="ndarray"),
            IOSpec(name="n_nodes", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="has_cycle", type_desc="bool"),
            IOSpec(name="cycle_nodes", type_desc="ndarray"),
        ],
        description="Detect cycles in directed graph via iterative DFS back-edge detection.",
        type_signature="ndarray, int -> tuple[bool, ndarray]",
    )
    rhs = CDGExport(
        nodes=[src, cycle_check, init],
        edges=[
            _edge("src", "cycle_check"),
            _edge("cycle_check", "init"),
        ],
    )

    return RewriteRule(
        name="insert_cycle_detection_before_init",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        priority=3,
    )


def _build_insert_connectivity_check() -> RewriteRule:
    """Interpose ``check_connectivity`` before Init Visited.

    When the graph is disconnected, a single-source traversal will miss
    entire components.  This pre-check labels components so the traversal
    can be run per-component or the user is warned.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    init = _node(
        "init",
        _INIT_VISITED,
        ConceptType.GRAPH_TRAVERSAL,
    )
    lhs = CDGExport(nodes=[src, init], edges=[_edge("src", "init")])
    interface = CDGExport(nodes=[src, init], edges=[])

    conn_check = _node(
        "conn_check",
        "Check Connectivity",
        ConceptType.GRAPH_TRAVERSAL,
        matched_primitive="check_connectivity",
        inputs=[
            IOSpec(name="adjacency", type_desc="ndarray"),
            IOSpec(name="n_nodes", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="n_components", type_desc="int"),
            IOSpec(name="component_labels", type_desc="ndarray"),
        ],
        description="Label connected components on undirected view of directed graph.",
        type_signature="ndarray, int -> tuple[int, ndarray]",
    )
    rhs = CDGExport(
        nodes=[src, conn_check, init],
        edges=[
            _edge("src", "conn_check"),
            _edge("conn_check", "init"),
        ],
    )

    return RewriteRule(
        name="insert_connectivity_check_before_init",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        priority=2,
    )


def _build_insert_frontier_overflow_detection() -> RewriteRule:
    """Interpose ``detect_frontier_overflow`` after Update Frontier.

    An anomalously large frontier suggests redundant node re-expansion
    (missing visited check) or pathological graph structure.
    """
    update = _node(
        "update",
        _UPDATE_FRONTIER,
        ConceptType.GRAPH_TRAVERSAL,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    overflow = _node(
        "overflow",
        "Detect Frontier Overflow",
        ConceptType.GRAPH_TRAVERSAL,
        matched_primitive="detect_frontier_overflow",
        inputs=[
            IOSpec(name="frontier_sizes", type_desc="ndarray"),
            IOSpec(name="n_nodes", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="overflow_mask", type_desc="ndarray"),
            IOSpec(name="max_frontier_size", type_desc="int"),
        ],
        description="Flag iterations where frontier size exceeds sqrt(n_nodes).",
        type_signature="ndarray, int -> tuple[ndarray, int]",
    )
    rhs = CDGExport(
        nodes=[update, overflow, sink],
        edges=[
            _edge("update", "overflow"),
            _edge("overflow", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_frontier_overflow_detection_after_update",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_visited_compaction() -> RewriteRule:
    """Interpose ``compact_visited_set`` after Update Frontier.

    When visited ratio is high, switching from sparse index list to dense
    bitmask is more cache-friendly.
    """
    update = _node(
        "update",
        _UPDATE_FRONTIER,
        ConceptType.GRAPH_TRAVERSAL,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    compact = _node(
        "compact",
        "Compact Visited Set",
        ConceptType.GRAPH_TRAVERSAL,
        matched_primitive="compact_visited_set",
        inputs=[
            IOSpec(name="visited_indices", type_desc="ndarray"),
            IOSpec(name="n_nodes", type_desc="int"),
        ],
        outputs=[IOSpec(name="compact_visited", type_desc="ndarray")],
        description="Convert sparse visited-index list to dense boolean bitmask.",
        type_signature="ndarray, int -> ndarray",
    )
    rhs = CDGExport(
        nodes=[update, compact, sink],
        edges=[
            _edge("update", "compact"),
            _edge("compact", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_visited_compaction_after_update",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_cycles(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect cycles in the input graph."""
    intermediates = context.intermediates or {}
    adjacency = intermediates.get("adjacency")
    n_nodes = intermediates.get("n_nodes")

    if adjacency is None or n_nodes is None:
        return None

    try:
        adjacency = np.asarray(adjacency, dtype=np.int64)
        n = int(n_nodes)
    except (ValueError, TypeError):
        return None

    if n == 0 or len(adjacency) == 0:
        return None

    # Build adjacency list and run DFS cycle detection inline
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(len(adjacency)):
        src, tgt = int(adjacency[i, 0]), int(adjacency[i, 1])
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)

    color = np.full(n, _WHITE, dtype=np.int8)
    has_cycle = False

    for start in range(n):
        if color[start] != _WHITE:
            continue
        stack: list[tuple[int, int]] = [(start, 0)]
        color[start] = _GRAY
        while stack:
            node, ni = stack[-1]
            if ni < len(adj[node]):
                stack[-1] = (node, ni + 1)
                neighbor = adj[node][ni]
                if color[neighbor] == _GRAY:
                    has_cycle = True
                    break
                elif color[neighbor] == _WHITE:
                    color[neighbor] = _GRAY
                    stack.append((neighbor, 0))
            else:
                color[node] = _BLACK
                stack.pop()
        if has_cycle:
            break

    if has_cycle:
        return ExpansionDiagnostic(
            rule_name="insert_cycle_detection_before_init",
            severity=1.0,
            evidence="Cycle detected in directed graph",
            metric_name="has_cycle",
            metric_value=1.0,
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_connectivity(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect disconnected components in the graph."""
    intermediates = context.intermediates or {}
    adjacency = intermediates.get("adjacency")
    n_nodes = intermediates.get("n_nodes")

    if adjacency is None or n_nodes is None:
        return None

    try:
        adjacency = np.asarray(adjacency, dtype=np.int64)
        n = int(n_nodes)
    except (ValueError, TypeError):
        return None

    if n <= 1:
        return None

    # Build undirected adjacency list
    from collections import deque

    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(len(adjacency)):
        src, tgt = int(adjacency[i, 0]), int(adjacency[i, 1])
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)
            adj[tgt].append(src)

    visited = np.zeros(n, dtype=bool)
    n_components = 0
    for start in range(n):
        if visited[start]:
            continue
        queue = deque([start])
        visited[start] = True
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)
        n_components += 1

    if n_components > 1:
        return ExpansionDiagnostic(
            rule_name="insert_connectivity_check_before_init",
            severity=min(1.0, n_components / 10.0),
            evidence=(
                f"Graph has {n_components} connected components "
                f"(single-source traversal will miss {n_components - 1})"
            ),
            metric_name="n_components",
            metric_value=float(n_components),
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_frontier_overflow(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect anomalously large frontier sizes."""
    intermediates = context.intermediates or {}
    frontier_sizes = intermediates.get("frontier_sizes")
    n_nodes = intermediates.get("n_nodes")

    if frontier_sizes is None or n_nodes is None:
        return None

    try:
        frontier_sizes = np.asarray(frontier_sizes, dtype=np.int64)
        n = int(n_nodes)
    except (ValueError, TypeError):
        return None

    if len(frontier_sizes) == 0 or n <= 0:
        return None

    threshold = np.sqrt(max(n, 1))
    max_frontier = float(np.max(frontier_sizes))
    ratio = max_frontier / threshold

    if ratio > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_frontier_overflow_detection_after_update",
            severity=min(1.0, ratio / 5.0),
            evidence=(
                f"Max frontier size {int(max_frontier)} exceeds "
                f"sqrt({n})={threshold:.1f} by {ratio:.1f}x"
            ),
            metric_name="frontier_overflow_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_visited_density(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect high visited ratio suggesting bitmask compaction."""
    intermediates = context.intermediates or {}
    visited_ratio = intermediates.get("visited_ratio")

    if visited_ratio is None:
        return None

    try:
        ratio = float(visited_ratio)
    except (ValueError, TypeError):
        return None

    if ratio > 0.8:
        return ExpansionDiagnostic(
            rule_name="insert_visited_compaction_after_update",
            severity=min(1.0, (ratio - 0.8) / 0.2),
            evidence=(
                f"Visited ratio {ratio:.2f} exceeds 0.8 threshold "
                f"— dense bitmask is more cache-friendly"
            ),
            metric_name="visited_ratio",
            metric_value=ratio,
            threshold=0.8,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class GraphTraversalExpansionRuleSet:
    """Expansion rules for graph traversal pipelines (BFS, DFS, topological sort)."""

    name = "graph_traversal"
    domain = "graph_traversal"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_cycle_detection(),
            _build_insert_connectivity_check(),
            _build_insert_frontier_overflow_detection(),
            _build_insert_visited_compaction(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        cyc = _diagnose_cycles(cdg, context)
        if cyc is not None:
            diagnostics.append(cyc)

        conn = _diagnose_connectivity(cdg, context)
        if conn is not None:
            diagnostics.append(conn)

        frontier = _diagnose_frontier_overflow(cdg, context)
        if frontier is not None:
            diagnostics.append(frontier)

        visited = _diagnose_visited_density(cdg, context)
        if visited is not None:
            diagnostics.append(visited)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

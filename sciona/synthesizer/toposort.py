"""Topological sort of CDG nodes by data-flow edges."""

from __future__ import annotations

from collections import deque

from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge


def toposort_nodes(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> list[str]:
    """Return node IDs in dependency order (leaves first, root last).

    Uses Kahn's algorithm. Raises ValueError on cycles.
    """
    node_ids = {n.node_id for n in nodes}

    # Build adjacency: source -> [targets]
    # and in-degree counts (only for edges between nodes in our set)
    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    successors: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in edges:
        if edge.source_id in node_ids and edge.target_id in node_ids:
            successors[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

    # Seed the queue with nodes that have no incoming edges
    queue: deque[str] = deque()
    for nid in node_ids:
        if in_degree[nid] == 0:
            queue.append(nid)

    result: list[str] = []
    while queue:
        nid = queue.popleft()
        result.append(nid)
        for succ in successors[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(result) != len(node_ids):
        raise ValueError(
            f"Cycle detected: sorted {len(result)} of {len(node_ids)} nodes"
        )

    return result


def detect_cycle_partition(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> tuple[list[str], set[str], bool]:
    """Partition nodes into acyclic and cyclic sets, checking for valid cycles.

    Runs Kahn's algorithm to find all nodes reachable acyclically.  Remaining
    nodes form the cycle set.  A cycle is considered *valid* when every node
    in it has ``concept_type`` equal to ``MESSAGE_PASSING`` or ``FIXED_POINT``.

    Returns:
        (acyclic_sorted_ids, cycle_node_ids, is_valid_cycle)

        *acyclic_sorted_ids* — topologically sorted IDs of nodes **not** in
        any cycle.
        *cycle_node_ids* — the set of node IDs that participate in a cycle.
        *is_valid_cycle* — ``True`` when ``cycle_node_ids`` is non-empty and
        all cycle nodes are MESSAGE_PASSING or FIXED_POINT.
    """
    node_ids = {n.node_id for n in nodes}
    node_map = {n.node_id: n for n in nodes}

    in_degree: dict[str, int] = {nid: 0 for nid in node_ids}
    successors: dict[str, list[str]] = {nid: [] for nid in node_ids}

    for edge in edges:
        if edge.source_id in node_ids and edge.target_id in node_ids:
            successors[edge.source_id].append(edge.target_id)
            in_degree[edge.target_id] += 1

    queue: deque[str] = deque()
    for nid in node_ids:
        if in_degree[nid] == 0:
            queue.append(nid)

    acyclic: list[str] = []
    while queue:
        nid = queue.popleft()
        acyclic.append(nid)
        for succ in successors[nid]:
            in_degree[succ] -= 1
            if in_degree[succ] == 0:
                queue.append(succ)

    if len(acyclic) == len(node_ids):
        return acyclic, set(), False

    cycle_ids = node_ids - set(acyclic)

    _VALID_CYCLE_TYPES = {ConceptType.MESSAGE_PASSING, ConceptType.FIXED_POINT}
    is_valid = all(
        getattr(node_map.get(nid), "concept_type", None) in _VALID_CYCLE_TYPES
        for nid in cycle_ids
        if nid in node_map
    )

    return acyclic, cycle_ids, is_valid


def toposort_with_fixed_points(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> tuple[list[str], dict[str, list[str]]]:
    """Topological sort that treats FIXED_POINT subtrees as opaque.

    1. Identifies FIXED_POINT nodes and their children.
    2. Topologically sorts each FIXED_POINT body independently (must be
       acyclic internally).
    3. Topologically sorts the top-level graph treating FIXED_POINT nodes
       as atomic (opaque).

    Returns:
        (top_level_order, fixed_point_bodies)

        *top_level_order* — node IDs in topological order at the top level.
        FIXED_POINT parent nodes appear as single entries.
        *fixed_point_bodies* — mapping from FIXED_POINT node_id to the
        topologically sorted list of child IDs.
    """
    node_map = {n.node_id: n for n in nodes}

    # Identify FIXED_POINT nodes and their children
    fp_nodes: dict[str, AlgorithmicNode] = {}
    fp_children: dict[str, set[str]] = {}
    for n in nodes:
        if n.concept_type == ConceptType.FIXED_POINT:
            fp_nodes[n.node_id] = n
            fp_children[n.node_id] = set(n.children) if n.children else set()

    # Flatten all children belonging to any fixed-point
    all_fp_child_ids: set[str] = set()
    for children in fp_children.values():
        all_fp_child_ids |= children

    # Sort each FIXED_POINT body independently
    fixed_point_bodies: dict[str, list[str]] = {}
    for fp_id, child_ids in fp_children.items():
        body_nodes = [node_map[cid] for cid in child_ids if cid in node_map]
        # Edges internal to the body
        body_edges = [
            e for e in edges
            if e.source_id in child_ids and e.target_id in child_ids
        ]
        if body_nodes:
            fixed_point_bodies[fp_id] = toposort_nodes(body_nodes, body_edges)
        else:
            fixed_point_bodies[fp_id] = []

    # Top-level sort: exclude FP children (they are inside opaque FP nodes)
    top_nodes = [n for n in nodes if n.node_id not in all_fp_child_ids]
    top_edges = [
        e for e in edges
        if e.source_id not in all_fp_child_ids
        and e.target_id not in all_fp_child_ids
    ]
    top_order = toposort_nodes(top_nodes, top_edges)

    return top_order, fixed_point_bodies

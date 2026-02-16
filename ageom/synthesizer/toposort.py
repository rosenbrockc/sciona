"""Topological sort of CDG nodes by data-flow edges."""

from __future__ import annotations

from collections import deque

from ageom.architect.models import AlgorithmicNode, DependencyEdge


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

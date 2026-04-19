"""Cycle detection and deterministic cycle-breaking for CDGs.

The cycle detector uses Kahn's algorithm (topological sort). The cycle
breaker applies deterministic patches for common patterns: unbounded
loops, missing convergence checks, and undamped message updates.
"""

from __future__ import annotations

from collections import deque


def detect_cycles(
    node_ids: set[str],
    edges: list[tuple[str, str]],
) -> set[str]:
    """Detect cyclic nodes in a directed graph using Kahn's algorithm.

    Args:
        node_ids: Set of all node identifiers.
        edges: List of (source_id, target_id) directed edges.

    Returns:
        Set of node IDs that participate in cycles. Empty if the graph
        is acyclic.
    """
    adjacency: dict[str, set[str]] = {nid: set() for nid in node_ids}
    indegree: dict[str, int] = {nid: 0 for nid in node_ids}

    for source_id, target_id in edges:
        if source_id not in node_ids or target_id not in node_ids:
            continue
        if target_id in adjacency[source_id]:
            continue
        adjacency[source_id].add(target_id)
        indegree[target_id] += 1

    queue = deque(nid for nid, deg in indegree.items() if deg == 0)
    processed = 0
    while queue:
        nid = queue.popleft()
        processed += 1
        for next_id in adjacency[nid]:
            indegree[next_id] -= 1
            if indegree[next_id] == 0:
                queue.append(next_id)

    if processed == len(node_ids):
        return set()
    return {nid for nid, deg in indegree.items() if deg > 0}


def break_cycle(
    deadlock_nodes: list[str],
    cycle_edges: list[str],
    witness_source: str,
) -> tuple[list[dict[str, object]], str] | None:
    """Attempt deterministic cycle-breaking patches on witness source.

    Tries three strategies in order:
    1. Iteration cap: converts ``while True`` to bounded loop.
    2. Convergence check: replaces ``converged = False`` with a real check.
    3. Message damping: inserts ``0.5 * new + 0.5 * old`` damping.

    Args:
        deadlock_nodes: Node IDs involved in the cycle.
        cycle_edges: String descriptions of cycle edges.
        witness_source: Current witness Python source code.

    Returns:
        A (patches, strategy_name) tuple if a fix was found, or None
        if no deterministic fix applies.
    """
    from sciona.ingester.deterministic_cycle_breaker import _break_cycle

    return _break_cycle(deadlock_nodes, cycle_edges, witness_source)


__all__ = ["detect_cycles", "break_cycle"]

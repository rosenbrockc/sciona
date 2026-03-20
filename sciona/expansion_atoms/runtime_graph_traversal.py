"""Runtime atoms for Graph Traversal expansion rules.

Provides deterministic, pure functions for graph traversal
quality diagnostics and structural pre-checks:

  - Cycle detection (iterative DFS with back-edge detection)
  - Connectivity analysis (BFS-based component labeling)
  - Visited-set compaction (sparse → dense bitmask)
  - Frontier overflow detection (anomalous queue/stack growth)
"""

from __future__ import annotations

from collections import deque

import numpy as np


# ---------------------------------------------------------------------------
# Cycle detection
# ---------------------------------------------------------------------------

# DFS color states
_WHITE, _GRAY, _BLACK = 0, 1, 2


def detect_cycles(
    adjacency: np.ndarray,
    n_nodes: int,
) -> tuple[bool, np.ndarray]:
    """Detect cycles in a directed graph via iterative DFS back-edge detection.

    Uses WHITE/GRAY/BLACK coloring.  A back-edge (to a GRAY node) indicates
    a cycle.  All nodes reachable from any back-edge target via the GRAY
    path are reported as cycle participants.

    Args:
        adjacency: shape (n_edges, 2) — each row is (source, target).
        n_nodes: number of nodes in the graph.

    Returns:
        (has_cycle, cycle_nodes) where cycle_nodes contains the indices
        of nodes participating in at least one cycle (empty if acyclic).
    """
    adjacency = np.asarray(adjacency, dtype=np.int64)
    n = int(n_nodes)

    if n == 0 or len(adjacency) == 0:
        return False, np.empty(0, dtype=np.int64)

    # Build adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(len(adjacency)):
        src, tgt = int(adjacency[i, 0]), int(adjacency[i, 1])
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)

    color = np.full(n, _WHITE, dtype=np.int8)
    parent = np.full(n, -1, dtype=np.int64)
    cycle_nodes_set: set[int] = set()

    for start in range(n):
        if color[start] != _WHITE:
            continue

        # Iterative DFS using explicit stack
        # Stack entries: (node, neighbor_index)
        stack: list[tuple[int, int]] = [(start, 0)]
        color[start] = _GRAY

        while stack:
            node, ni = stack[-1]
            if ni < len(adj[node]):
                stack[-1] = (node, ni + 1)
                neighbor = adj[node][ni]
                if color[neighbor] == _GRAY:
                    # Back-edge found → collect cycle nodes from stack
                    in_cycle = False
                    for sn, _ in stack:
                        if sn == neighbor:
                            in_cycle = True
                        if in_cycle:
                            cycle_nodes_set.add(sn)
                elif color[neighbor] == _WHITE:
                    color[neighbor] = _GRAY
                    parent[neighbor] = node
                    stack.append((neighbor, 0))
            else:
                color[node] = _BLACK
                stack.pop()

    if cycle_nodes_set:
        return True, np.array(sorted(cycle_nodes_set), dtype=np.int64)
    return False, np.empty(0, dtype=np.int64)


# ---------------------------------------------------------------------------
# Connectivity analysis
# ---------------------------------------------------------------------------


def check_connectivity(
    adjacency: np.ndarray,
    n_nodes: int,
) -> tuple[int, np.ndarray]:
    """Label connected components on the undirected view of a directed graph.

    Treats each directed edge as bidirectional for connectivity purposes.
    Uses BFS-based flood fill.

    Args:
        adjacency: shape (n_edges, 2) — each row is (source, target).
        n_nodes: number of nodes in the graph.

    Returns:
        (n_components, component_labels) where component_labels[i] is the
        component index of node i (0-indexed).
    """
    adjacency = np.asarray(adjacency, dtype=np.int64)
    n = int(n_nodes)

    if n == 0:
        return 0, np.empty(0, dtype=np.int64)

    # Build undirected adjacency list
    adj: list[list[int]] = [[] for _ in range(n)]
    for i in range(len(adjacency)):
        src, tgt = int(adjacency[i, 0]), int(adjacency[i, 1])
        if 0 <= src < n and 0 <= tgt < n:
            adj[src].append(tgt)
            adj[tgt].append(src)

    labels = np.full(n, -1, dtype=np.int64)
    component = 0

    for start in range(n):
        if labels[start] >= 0:
            continue
        # BFS flood fill
        queue = deque([start])
        labels[start] = component
        while queue:
            node = queue.popleft()
            for neighbor in adj[node]:
                if labels[neighbor] < 0:
                    labels[neighbor] = component
                    queue.append(neighbor)
        component += 1

    return component, labels


# ---------------------------------------------------------------------------
# Visited-set compaction
# ---------------------------------------------------------------------------


def compact_visited_set(
    visited_indices: np.ndarray,
    n_nodes: int,
) -> np.ndarray:
    """Convert sparse visited-index list to dense boolean bitmask.

    When the visited ratio is high, a dense boolean array is more
    cache-friendly than a hash set of indices.  This is a pure
    representation change with no semantic difference.

    Args:
        visited_indices: 1-D array of visited node indices.
        n_nodes: total number of nodes in the graph.

    Returns:
        compact_visited — bool array of length n_nodes.
    """
    visited_indices = np.asarray(visited_indices, dtype=np.int64)
    n = int(n_nodes)

    compact = np.zeros(n, dtype=bool)
    valid = visited_indices[(visited_indices >= 0) & (visited_indices < n)]
    compact[valid] = True
    return compact


# ---------------------------------------------------------------------------
# Frontier overflow detection
# ---------------------------------------------------------------------------


def detect_frontier_overflow(
    frontier_sizes: np.ndarray,
    n_nodes: int,
) -> tuple[np.ndarray, int]:
    """Flag iterations where frontier size exceeds sqrt(n_nodes).

    An anomalously large frontier suggests redundant node re-expansion
    (missing visited check) or pathological graph structure.

    Args:
        frontier_sizes: 1-D array of frontier sizes per iteration.
        n_nodes: total number of nodes in the graph.

    Returns:
        (overflow_mask, max_frontier_size) where overflow_mask[k] is True
        if frontier_sizes[k] > sqrt(n_nodes).
    """
    frontier_sizes = np.asarray(frontier_sizes, dtype=np.int64)
    n = int(n_nodes)

    threshold = int(np.ceil(np.sqrt(max(n, 1))))
    overflow_mask = frontier_sizes > threshold
    max_frontier = int(np.max(frontier_sizes)) if len(frontier_sizes) > 0 else 0
    return overflow_mask, max_frontier

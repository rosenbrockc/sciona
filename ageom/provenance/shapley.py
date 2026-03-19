"""Shapley value computation for atom dependency DAGs."""

from __future__ import annotations

from collections import deque
from fractions import Fraction


def compute_shapley_values(
    dag: dict[str, set[str]],
) -> dict[str, Fraction]:
    """Compute Shapley values for atoms in a dependency DAG.

    Characteristic function: v(S) = 1 iff S contains **all** atoms
    in the transitive closure from the root(s).  Under this "all required"
    model every atom is equally pivotal, yielding uniform
    ``Fraction(1, n)`` for *n* atoms.

    Uses exact arithmetic (:class:`fractions.Fraction`).

    Post-condition
    --------------
    ``sum(values) == Fraction(1)``

    Raises
    ------
    ValueError
        If *dag* is empty or contains a cycle.
    """
    if not dag:
        raise ValueError("Cannot compute Shapley values on an empty graph.")

    # Validate acyclicity via topological sort.
    _topological_sort(dag)

    # Collect all nodes (keys and values).
    all_nodes: set[str] = set(dag.keys())
    for deps in dag.values():
        all_nodes.update(deps)

    n = len(all_nodes)
    value = Fraction(1, n)
    result = {node: value for node in all_nodes}

    assert sum(result.values()) == Fraction(1)
    return result


def _topological_sort(dag: dict[str, set[str]]) -> list[str]:
    """Kahn's algorithm.  Raises :exc:`ValueError` on cycle."""
    # Build full adjacency and in-degree from the DAG.
    all_nodes: set[str] = set(dag.keys())
    for deps in dag.values():
        all_nodes.update(deps)

    in_degree: dict[str, int] = {node: 0 for node in all_nodes}
    adjacency: dict[str, list[str]] = {node: [] for node in all_nodes}

    for node, deps in dag.items():
        for dep in deps:
            adjacency[dep].append(node)
            in_degree[node] += 1

    queue: deque[str] = deque(n for n in all_nodes if in_degree[n] == 0)
    order: list[str] = []

    while queue:
        current = queue.popleft()
        order.append(current)
        for neighbour in adjacency[current]:
            in_degree[neighbour] -= 1
            if in_degree[neighbour] == 0:
                queue.append(neighbour)

    if len(order) != len(all_nodes):
        raise ValueError("Cycle detected in dependency DAG.")

    return order


def _transitive_closure(dag: dict[str, set[str]], node: str) -> set[str]:
    """All transitive dependencies of *node* (inclusive)."""
    visited: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current in visited:
            continue
        visited.add(current)
        stack.extend(dag.get(current, set()))
    return visited

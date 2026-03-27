"""LLM pre-screen gate for CDG submissions.

Stateless filter that rejects obviously invalid or malicious CDG submissions
before consuming sandbox compute.  Three checks run sequentially (fail-fast):

1. Suspicious pattern scan (AST-based)
2. Structural validity (DAG acyclicity, atom existence)
3. Resource estimation (node/edge counts, tier assignment)
"""

from __future__ import annotations

import ast
from typing import Sequence

from sciona.clearinghouse.models import PreScreenResult

# ---------------------------------------------------------------------------
# Suspicious imports and calls
# ---------------------------------------------------------------------------

_BANNED_IMPORTS: frozenset[str] = frozenset({
    "socket",
    "urllib",
    "requests",
    "http",
    "subprocess",
    "ctypes",
    "cffi",
    "importlib",
})

_BANNED_CALL_NAMES: frozenset[str] = frozenset({
    "exec",
    "eval",
    "compile",
    "__import__",
})

_BANNED_ATTR_CALLS: frozenset[str] = frozenset({
    "os.system",
    "os.popen",
    "os.exec",
    "os.execv",
    "os.execvp",
    "os.execvpe",
})

_WRITE_MODES: frozenset[str] = frozenset({"w", "wb", "a", "ab", "x", "xb", "r+", "rb+"})


# ---------------------------------------------------------------------------
# AST scanning
# ---------------------------------------------------------------------------


def _scan_source(source: str) -> list[str]:
    """Scan a single source string for suspicious patterns.

    Returns a list of human-readable rejection reasons.
    """
    reasons: list[str] = []

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        reasons.append(f"SyntaxError: {exc}")
        return reasons

    for node in ast.walk(tree):
        # --- imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top in _BANNED_IMPORTS:
                    reasons.append(f"Banned import: {alias.name}")

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top = node.module.split(".")[0]
                if top in _BANNED_IMPORTS:
                    reasons.append(f"Banned import: {node.module}")

        # --- calls ---
        elif isinstance(node, ast.Call):
            # Direct call: exec(...), eval(...)
            if isinstance(node.func, ast.Name):
                if node.func.id in _BANNED_CALL_NAMES:
                    reasons.append(f"Banned call: {node.func.id}()")

            # Attribute call: os.system(...)
            elif isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name):
                    full = f"{node.func.value.id}.{node.func.attr}"
                    if full in _BANNED_ATTR_CALLS:
                        reasons.append(f"Banned call: {full}()")

                    # open() with write mode
                    if full == "builtins.open" or (
                        isinstance(node.func.value, ast.Name)
                        and node.func.value.id == "builtins"
                        and node.func.attr == "open"
                    ):
                        _check_open_write(node, reasons)

            # Top-level open() with write mode
            if isinstance(node.func, ast.Name) and node.func.id == "open":
                _check_open_write(node, reasons)

            # pathlib write methods
            if isinstance(node.func, ast.Attribute) and node.func.attr in (
                "write_text",
                "write_bytes",
            ):
                reasons.append(f"Filesystem write: .{node.func.attr}()")

    return reasons


def _check_open_write(node: ast.Call, reasons: list[str]) -> None:
    """Check if an open() call uses a write mode."""
    # Check positional mode argument (second arg)
    if len(node.args) >= 2 and isinstance(node.args[1], ast.Constant):
        if isinstance(node.args[1].value, str) and node.args[1].value in _WRITE_MODES:
            reasons.append(f"Filesystem write: open(..., {node.args[1].value!r})")
    # Check keyword mode argument
    for kw in node.keywords:
        if kw.arg == "mode" and isinstance(kw.value, ast.Constant):
            if isinstance(kw.value.value, str) and kw.value.value in _WRITE_MODES:
                reasons.append(f"Filesystem write: open(..., mode={kw.value.value!r})")


# ---------------------------------------------------------------------------
# Structural validity
# ---------------------------------------------------------------------------


def _check_structure(
    atom_fqdns: Sequence[str],
    dependency_edges: Sequence[tuple[str, str]],
    known_atoms: frozenset[str] | None = None,
    fixed_point_node_ids: frozenset[str] | None = None,
) -> list[str]:
    """Check structural validity of a CDG.

    Parameters
    ----------
    atom_fqdns
        All atom FQDNs in the CDG.
    dependency_edges
        Directed edges (from, to) in the dependency DAG.
    known_atoms
        Set of FQDNs registered in the global registry.  If ``None``,
        skip the registry existence check.
    fixed_point_node_ids
        Set of node FQDNs that belong to FIXED_POINT-annotated subgraphs.
        Cycles among these nodes are permitted and will not trigger a
        rejection.
    """
    reasons: list[str] = []
    _fp_ids = fixed_point_node_ids or frozenset()

    if known_atoms is not None:
        for fqdn in atom_fqdns:
            if fqdn not in known_atoms:
                reasons.append(f"Unknown atom: {fqdn}")

    # DAG acyclicity check via Kahn's algorithm.
    in_degree: dict[str, int] = {fqdn: 0 for fqdn in atom_fqdns}
    adjacency: dict[str, list[str]] = {fqdn: [] for fqdn in atom_fqdns}
    for src, dst in dependency_edges:
        adjacency.setdefault(src, []).append(dst)
        in_degree.setdefault(dst, 0)
        in_degree[dst] = in_degree.get(dst, 0) + 1

    queue = [n for n, d in in_degree.items() if d == 0]
    visited = 0
    while queue:
        current = queue.pop()
        visited += 1
        for nb in adjacency.get(current, []):
            in_degree[nb] -= 1
            if in_degree[nb] == 0:
                queue.append(nb)

    if visited != len(in_degree):
        # Some nodes are in a cycle — check if ALL cycle nodes are in
        # FIXED_POINT subgraphs.
        cycle_nodes = {
            nid for nid, deg in in_degree.items() if deg > 0
        }
        if not cycle_nodes.issubset(_fp_ids):
            reasons.append("Dependency graph contains a cycle")

    return reasons


# ---------------------------------------------------------------------------
# Resource estimation
# ---------------------------------------------------------------------------

# Default limits
DEFAULT_MAX_NODES = 100
DEFAULT_MAX_DEPTH = 20


def _estimate_resources(
    node_count: int,
    max_depth: int,
    *,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_depth_limit: int = DEFAULT_MAX_DEPTH,
) -> tuple[list[str], str, float, float]:
    """Estimate resource requirements and flag exceedances.

    Returns (reasons, tier, memory_gb, runtime_minutes).
    """
    reasons: list[str] = []

    if node_count > max_nodes:
        reasons.append(
            f"Node count {node_count} exceeds limit {max_nodes}"
        )

    if max_depth > max_depth_limit:
        reasons.append(
            f"DAG depth {max_depth} exceeds limit {max_depth_limit}"
        )

    # Heuristic tier assignment
    if node_count <= 20:
        tier = "standard"
        memory_gb = 4.0
        runtime_min = 5.0
    elif node_count <= 60:
        tier = "heavy"
        memory_gb = 16.0
        runtime_min = 30.0
    else:
        tier = "gpu"
        memory_gb = 64.0
        runtime_min = 120.0

    return reasons, tier, memory_gb, runtime_min


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def prescreen(
    atom_sources: dict[str, str],
    dependency_edges: list[tuple[str, str]] | None = None,
    *,
    known_atoms: frozenset[str] | None = None,
    max_nodes: int = DEFAULT_MAX_NODES,
    max_depth: int = DEFAULT_MAX_DEPTH,
    fixed_point_node_ids: frozenset[str] | None = None,
) -> PreScreenResult:
    """Run the pre-screen gate on a CDG submission.

    Parameters
    ----------
    atom_sources
        Mapping of atom FQDN to Python source code.
    dependency_edges
        Directed edges in the CDG dependency graph.
    known_atoms
        Registry of known atom FQDNs for existence check.
    max_nodes
        Maximum allowed node count.
    max_depth
        Maximum allowed DAG depth.
    fixed_point_node_ids
        Node FQDNs belonging to FIXED_POINT subgraphs.  Cycles among
        these nodes are tolerated during the structural check.

    Returns
    -------
    PreScreenResult
        Contains pass/reject verdict with reasons and resource estimates.
    """
    reasons: list[str] = []

    # Phase 1: Suspicious pattern scan
    for fqdn, source in atom_sources.items():
        for reason in _scan_source(source):
            reasons.append(f"[{fqdn}] {reason}")

    if reasons:
        return PreScreenResult(
            passed=False,
            rejection_reasons=reasons,
            estimated_tier="standard",
        )

    # Phase 2: Structural validity
    edges = dependency_edges or []
    struct_reasons = _check_structure(
        list(atom_sources.keys()),
        edges,
        known_atoms=known_atoms,
        fixed_point_node_ids=fixed_point_node_ids,
    )
    reasons.extend(struct_reasons)

    if reasons:
        return PreScreenResult(
            passed=False,
            rejection_reasons=reasons,
            estimated_tier="standard",
        )

    # Phase 3: Resource estimation
    # Compute max depth via longest path in DAG
    fqdns = list(atom_sources.keys())
    adjacency: dict[str, list[str]] = {f: [] for f in fqdns}
    for src, dst in edges:
        adjacency.setdefault(src, []).append(dst)

    depths: dict[str, int] = {}
    _visiting: set[str] = set()  # cycle guard

    def _depth(node: str) -> int:
        if node in depths:
            return depths[node]
        if node in _visiting:
            # Back-edge in a cycle (allowed for FP nodes) — stop recursion
            return 1
        _visiting.add(node)
        children = adjacency.get(node, [])
        d = 1 + max((_depth(c) for c in children), default=0) if children else 1
        _visiting.discard(node)
        depths[node] = d
        return d

    dag_depth = max((_depth(f) for f in fqdns), default=0) if fqdns else 0

    resource_reasons, tier, memory_gb, runtime_min = _estimate_resources(
        len(fqdns), dag_depth, max_nodes=max_nodes, max_depth_limit=max_depth,
    )
    reasons.extend(resource_reasons)

    return PreScreenResult(
        passed=len(reasons) == 0,
        rejection_reasons=reasons,
        estimated_tier=tier,
        estimated_memory_gb=memory_gb,
        estimated_runtime_minutes=runtime_min,
    )

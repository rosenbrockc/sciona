"""Registry for graph traversal primitives and expansion atoms."""

from __future__ import annotations

GRAPH_TRAVERSAL_DECLARATIONS = {
    # --- Expansion atoms (inserted by DPO rewriter) ---
    "detect_cycles": (
        "sciona.expansion_atoms.runtime_graph_traversal.detect_cycles",
        "np.ndarray, int -> tuple[bool, np.ndarray]",
        "Detect cycles in a directed graph via iterative DFS back-edge detection.",
    ),
    "check_connectivity": (
        "sciona.expansion_atoms.runtime_graph_traversal.check_connectivity",
        "np.ndarray, int -> tuple[int, np.ndarray]",
        "Label connected components on the undirected view of a directed graph.",
    ),
    "compact_visited_set": (
        "sciona.expansion_atoms.runtime_graph_traversal.compact_visited_set",
        "np.ndarray, int -> np.ndarray",
        "Convert sparse visited-index list to dense boolean bitmask.",
    ),
    "detect_frontier_overflow": (
        "sciona.expansion_atoms.runtime_graph_traversal.detect_frontier_overflow",
        "np.ndarray, int -> tuple[np.ndarray, int]",
        "Flag iterations where frontier size exceeds sqrt(n_nodes).",
    ),
}

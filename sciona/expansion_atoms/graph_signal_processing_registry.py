"""Registry for graph signal processing primitives and expansion atoms."""

from __future__ import annotations

GRAPH_SIGNAL_PROCESSING_DECLARATIONS = {
    "validate_graph_connectivity": (
        "sciona.expansion_atoms.runtime_graph_signal_processing.validate_graph_connectivity",
        "ndarray -> tuple[int, bool]",
        "Validate that the graph is connected.",
    ),
    "check_laplacian_symmetry": (
        "sciona.expansion_atoms.runtime_graph_signal_processing.check_laplacian_symmetry",
        "ndarray, float -> tuple[float, bool]",
        "Check that the graph Laplacian is symmetric.",
    ),
    "analyze_spectral_gap": (
        "sciona.expansion_atoms.runtime_graph_signal_processing.analyze_spectral_gap",
        "ndarray -> tuple[float, bool]",
        "Analyze the spectral gap (algebraic connectivity) of the graph.",
    ),
    "validate_filter_response": (
        "sciona.expansion_atoms.runtime_graph_signal_processing.validate_filter_response",
        "ndarray, ndarray -> tuple[float, bool]",
        "Validate that the graph filter response is well-behaved.",
    ),
}

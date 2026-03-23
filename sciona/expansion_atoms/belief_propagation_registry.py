"""Registry for belief propagation primitives and expansion atoms."""

from __future__ import annotations

BELIEF_PROPAGATION_DECLARATIONS = {
    "monitor_message_convergence": (
        "sciona.expansion_atoms.runtime_belief_propagation.monitor_message_convergence",
        "ndarray, float -> tuple[float, bool]",
        "Monitor convergence of message passing iterations.",
    ),
    "validate_belief_normalization": (
        "sciona.expansion_atoms.runtime_belief_propagation.validate_belief_normalization",
        "ndarray, float -> tuple[float, bool]",
        "Validate that beliefs (marginals) are properly normalized.",
    ),
    "analyze_message_damping": (
        "sciona.expansion_atoms.runtime_belief_propagation.analyze_message_damping",
        "ndarray -> tuple[float, bool]",
        "Analyze whether messages oscillate, suggesting damping is needed.",
    ),
    "detect_graph_cycles": (
        "sciona.expansion_atoms.runtime_belief_propagation.detect_graph_cycles",
        "ndarray -> tuple[int, bool]",
        "Detect cycles in the factor graph.",
    ),
}

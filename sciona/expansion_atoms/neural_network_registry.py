"""Registry for neural network primitives and expansion atoms."""

from __future__ import annotations

NEURAL_NETWORK_DECLARATIONS = {
    "detect_gradient_explosion": (
        "sciona.expansion_atoms.runtime_neural_network.detect_gradient_explosion",
        "ndarray -> tuple[float, bool]",
        "Detect exploding gradients by checking max gradient norm.",
    ),
    "analyze_activation_statistics": (
        "sciona.expansion_atoms.runtime_neural_network.analyze_activation_statistics",
        "ndarray -> tuple[float, bool]",
        "Analyze activation statistics to detect dead neurons.",
    ),
    "monitor_loss_convergence": (
        "sciona.expansion_atoms.runtime_neural_network.monitor_loss_convergence",
        "ndarray -> tuple[float, bool]",
        "Monitor loss convergence to detect plateaus.",
    ),
    "check_weight_distribution": (
        "sciona.expansion_atoms.runtime_neural_network.check_weight_distribution",
        "ndarray -> tuple[float, bool]",
        "Check weight distribution balance across layers.",
    ),
}

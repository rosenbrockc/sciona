"""Registry for baseline analysis primitives and expansion atoms."""

from __future__ import annotations

BASELINE_ANALYSIS_DECLARATIONS = {
    "check_onset_coverage": (
        "sciona.expansion_atoms.runtime_baseline_analysis.check_onset_coverage",
        "list, int -> tuple[float, bool]",
        "Check onset detection density relative to signal length.",
    ),
    "detect_padding_saturation": (
        "sciona.expansion_atoms.runtime_baseline_analysis.detect_padding_saturation",
        "ndarray, int -> tuple[float, bool]",
        "Detect excessive padding fraction in output signal.",
    ),
    "monitor_normalization_clipping": (
        "sciona.expansion_atoms.runtime_baseline_analysis.monitor_normalization_clipping",
        "ndarray -> tuple[float, bool]",
        "Monitor fraction of normalized values clipped at ceiling.",
    ),
    "validate_component_balance": (
        "sciona.expansion_atoms.runtime_baseline_analysis.validate_component_balance",
        "list[ndarray] -> tuple[float, bool]",
        "Validate energy balance across component contributions.",
    ),
}

"""Registry for dimensionality reduction primitives and expansion atoms."""

from __future__ import annotations

DIMENSIONALITY_REDUCTION_DECLARATIONS = {
    "analyze_explained_variance": (
        "sciona.expansion_atoms.runtime_dimensionality_reduction.analyze_explained_variance",
        "ndarray -> tuple[float, bool]",
        "Analyze cumulative explained variance ratio.",
    ),
    "detect_crowding": (
        "sciona.expansion_atoms.runtime_dimensionality_reduction.detect_crowding",
        "ndarray, ndarray -> tuple[float, bool]",
        "Detect crowding by measuring neighbor preservation quality.",
    ),
    "check_reconstruction_error": (
        "sciona.expansion_atoms.runtime_dimensionality_reduction.check_reconstruction_error",
        "ndarray, ndarray -> tuple[float, bool]",
        "Check reconstruction quality after dimensionality reduction.",
    ),
    "validate_orthogonality": (
        "sciona.expansion_atoms.runtime_dimensionality_reduction.validate_orthogonality",
        "ndarray -> tuple[float, bool]",
        "Validate orthogonality of projection components.",
    ),
}

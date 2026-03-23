"""Registry for geometry primitives and expansion atoms."""

from __future__ import annotations

GEOMETRY_DECLARATIONS = {
    "detect_collinear_points": (
        "sciona.expansion_atoms.runtime_geometry.detect_collinear_points",
        "ndarray, float -> tuple[int, float]",
        "Detect collinear point triples that cause degenerate geometry.",
    ),
    "analyze_numeric_precision": (
        "sciona.expansion_atoms.runtime_geometry.analyze_numeric_precision",
        "ndarray -> tuple[float, bool]",
        "Analyze whether point coordinates risk floating-point issues.",
    ),
    "detect_duplicate_points": (
        "sciona.expansion_atoms.runtime_geometry.detect_duplicate_points",
        "ndarray, float -> tuple[int, float]",
        "Detect duplicate or near-duplicate points.",
    ),
    "validate_convexity": (
        "sciona.expansion_atoms.runtime_geometry.validate_convexity",
        "ndarray -> tuple[int, bool]",
        "Validate that a polygon is convex by checking cross-product signs.",
    ),
}

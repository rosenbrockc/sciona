"""Registry for compression primitives and expansion atoms."""

from __future__ import annotations

COMPRESSION_DECLARATIONS = {
    "analyze_compression_ratio": (
        "sciona.expansion_atoms.runtime_compression.analyze_compression_ratio",
        "float, float, float -> tuple[float, bool]",
        "Compare achieved compression ratio to a theoretical entropy bound.",
    ),
    "validate_lossless_roundtrip": (
        "sciona.expansion_atoms.runtime_compression.validate_lossless_roundtrip",
        "ndarray, ndarray -> tuple[float, bool]",
        "Check whether decoding exactly reconstructs the original sequence.",
    ),
    "detect_dictionary_bloat": (
        "sciona.expansion_atoms.runtime_compression.detect_dictionary_bloat",
        "ndarray -> tuple[float, bool]",
        "Estimate dictionary growth relative to its initial size.",
    ),
    "monitor_encoding_throughput": (
        "sciona.expansion_atoms.runtime_compression.monitor_encoding_throughput",
        "ndarray, ndarray -> tuple[float, bool]",
        "Estimate symbols processed per millisecond.",
    ),
}

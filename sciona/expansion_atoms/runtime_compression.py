"""Runtime atoms for Compression expansion rules."""

from __future__ import annotations

import numpy as np


def analyze_compression_ratio(
    original_bits: float,
    compressed_bits: float,
    entropy_bound: float,
) -> tuple[float, bool]:
    """Compare achieved compression ratio to a theoretical entropy bound."""
    original = max(float(original_bits), 1.0)
    compressed = max(float(compressed_bits), 0.0)
    bound = max(float(entropy_bound), 0.0)
    achieved_ratio = compressed / original
    ratio_gap = max(achieved_ratio - bound, 0.0)
    return ratio_gap, ratio_gap <= 0.2


def validate_lossless_roundtrip(
    original: np.ndarray,
    decoded: np.ndarray,
) -> tuple[float, bool]:
    """Check whether decoding exactly reconstructs the original sequence."""
    lhs = np.asarray(original).ravel()
    rhs = np.asarray(decoded).ravel()
    if lhs.size == 0 and rhs.size == 0:
        return 0.0, True
    if lhs.size != rhs.size:
        return 1.0, False
    mismatch_fraction = float(np.mean(lhs != rhs))
    return mismatch_fraction, mismatch_fraction == 0.0


def detect_dictionary_bloat(
    dictionary_sizes: np.ndarray,
) -> tuple[float, bool]:
    """Estimate dictionary growth relative to its initial size."""
    sizes = np.asarray(dictionary_sizes, dtype=np.float64).ravel()
    if sizes.size < 2:
        return 1.0, True
    first = max(float(sizes[0]), 1.0)
    last = max(float(sizes[-1]), 0.0)
    growth_rate = last / first
    return growth_rate, growth_rate <= 2.0


def monitor_encoding_throughput(
    symbol_counts: np.ndarray,
    runtimes_ms: np.ndarray,
) -> tuple[float, bool]:
    """Estimate symbols processed per millisecond."""
    symbols = np.asarray(symbol_counts, dtype=np.float64).ravel()
    runtimes = np.asarray(runtimes_ms, dtype=np.float64).ravel()
    if symbols.size == 0 or runtimes.size == 0:
        return float("inf"), True
    total_symbols = float(np.sum(symbols))
    total_runtime = float(np.sum(runtimes))
    if total_runtime <= 0.0:
        return float("inf"), total_symbols > 0.0
    symbols_per_ms = total_symbols / total_runtime
    return symbols_per_ms, symbols_per_ms >= 1e3

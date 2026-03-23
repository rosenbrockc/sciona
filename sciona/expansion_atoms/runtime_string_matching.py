"""Runtime atoms for String Matching expansion rules.

Provides deterministic, pure functions for string matching
quality diagnostics and structural pre-checks:

  - Alphabet size analysis (detect large alphabets that affect table size)
  - Pattern-text ratio check (detect pathological length ratios)
  - Hash collision detection (Rabin-Karp spurious match rate)
  - Preprocessing table validation (failure function correctness)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Alphabet size analysis
# ---------------------------------------------------------------------------


def analyze_alphabet_size(
    text: np.ndarray,
    pattern: np.ndarray,
) -> tuple[int, int, float]:
    """Analyze alphabet sizes for algorithm selection guidance.

    Large alphabets make hash-based methods more effective (fewer
    collisions), while small alphabets favor DFA-based methods.

    Args:
        text: 1-D integer array of character codes from the text.
        pattern: 1-D integer array of character codes from the pattern.

    Returns:
        (text_alphabet_size, pattern_alphabet_size, overlap_ratio)
        where overlap_ratio is the fraction of pattern chars present
        in the text alphabet.
    """
    text = np.asarray(text, dtype=np.int64).ravel()
    pattern = np.asarray(pattern, dtype=np.int64).ravel()

    text_alpha = set(text.tolist()) if len(text) > 0 else set()
    pat_alpha = set(pattern.tolist()) if len(pattern) > 0 else set()

    if len(pat_alpha) == 0:
        return len(text_alpha), 0, 1.0

    overlap = len(pat_alpha & text_alpha) / len(pat_alpha)
    return len(text_alpha), len(pat_alpha), overlap


# ---------------------------------------------------------------------------
# Pattern-text ratio check
# ---------------------------------------------------------------------------


def check_pattern_text_ratio(
    pattern_length: int,
    text_length: int,
) -> tuple[float, str]:
    """Check the ratio of pattern length to text length.

    Very short patterns in very long texts may benefit from
    multi-pattern algorithms.  Pattern longer than text is an error.

    Args:
        pattern_length: length of the search pattern.
        text_length: length of the text to search.

    Returns:
        (ratio, assessment) where ratio is pattern_length / text_length
        and assessment is one of "error" (pattern > text), "short"
        (ratio < 0.01), "normal", or "long" (ratio > 0.5).
    """
    p = int(pattern_length)
    t = int(text_length)

    if t == 0:
        if p == 0:
            return 0.0, "normal"
        return float("inf"), "error"

    ratio = p / t

    if p > t:
        return ratio, "error"
    elif ratio < 0.01:
        return ratio, "short"
    elif ratio > 0.5:
        return ratio, "long"
    else:
        return ratio, "normal"


# ---------------------------------------------------------------------------
# Hash collision detection
# ---------------------------------------------------------------------------


def measure_hash_collision_rate(
    n_hash_matches: int,
    n_true_matches: int,
) -> tuple[float, bool]:
    """Measure the spurious match rate for Rabin-Karp style algorithms.

    When hash collisions are frequent, the verification step dominates
    and a deterministic algorithm (KMP) should be preferred.

    Args:
        n_hash_matches: total number of hash matches (true + spurious).
        n_true_matches: number of verified true matches.

    Returns:
        (collision_rate, is_excessive) where collision_rate is the
        fraction of hash matches that were spurious, and is_excessive
        is True if collision_rate > 0.5.
    """
    hm = int(n_hash_matches)
    tm = int(n_true_matches)

    if hm == 0:
        return 0.0, False

    spurious = max(0, hm - tm)
    rate = spurious / hm
    return rate, rate > 0.5


# ---------------------------------------------------------------------------
# Preprocessing table validation
# ---------------------------------------------------------------------------


def validate_failure_function(
    failure_table: np.ndarray,
    pattern_length: int,
) -> tuple[int, bool]:
    """Validate basic properties of a KMP failure function table.

    Checks that the failure function satisfies:
    1. failure[0] == 0 (or -1 depending on convention)
    2. failure[i] < i for all i
    3. No negative values (for 0-indexed convention)

    Args:
        failure_table: 1-D array of failure function values.
        pattern_length: expected length of the pattern.

    Returns:
        (n_violations, is_valid) where n_violations is the count of
        entries violating the failure function properties.
    """
    table = np.asarray(failure_table, dtype=np.int64).ravel()
    m = int(pattern_length)

    if len(table) == 0 or m == 0:
        return 0, True

    violations = 0

    # Check length matches
    if len(table) != m:
        violations += 1

    n = min(len(table), m)

    # Check failure[0] is 0 or -1
    if n > 0 and table[0] > 0:
        violations += 1

    # Check failure[i] < i and non-negative (for 0-indexed)
    for i in range(1, n):
        if table[i] >= i:
            violations += 1
        if table[i] < -1:
            violations += 1

    return violations, violations == 0

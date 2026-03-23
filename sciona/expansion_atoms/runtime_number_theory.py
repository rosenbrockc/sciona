"""Runtime atoms for Number Theory expansion rules.

Provides deterministic, pure functions for number theory algorithm
quality diagnostics and structural pre-checks:

  - Input range validation (overflow detection for modular arithmetic)
  - GCD convergence monitoring (detect slow Euclidean algorithm convergence)
  - Primality witness check (detect composite numbers skipping sieve)
  - Modular arithmetic overflow detection (intermediate overflow risk)
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Input range validation
# ---------------------------------------------------------------------------


def validate_input_range(
    values: np.ndarray,
    bit_width: int = 64,
) -> tuple[int, bool]:
    """Check whether input values are within safe computation range.

    Modular exponentiation and GCD computations can overflow if
    intermediate values exceed the integer width.

    Args:
        values: 1-D array of input integers.
        bit_width: target integer bit width (default 64).

    Returns:
        (n_overflow_risk, all_safe) where n_overflow_risk is the count
        of values where value² would exceed 2^bit_width.
    """
    vals = np.asarray(values, dtype=np.int64).ravel()

    if len(vals) == 0:
        return 0, True

    # value² overflow check: |v| > 2^(bit_width/2)
    threshold = 2 ** (bit_width // 2)
    n_risk = int(np.sum(np.abs(vals) > threshold))
    return n_risk, n_risk == 0


# ---------------------------------------------------------------------------
# GCD convergence monitoring
# ---------------------------------------------------------------------------


def monitor_gcd_convergence(
    remainders: np.ndarray,
) -> tuple[int, float]:
    """Monitor convergence rate of the Euclidean algorithm.

    The Euclidean algorithm converges when remainders decrease
    rapidly.  Slow convergence (Fibonacci-like inputs) causes
    maximum iteration count.

    Args:
        remainders: 1-D array of successive remainders in GCD computation.

    Returns:
        (n_steps, avg_reduction_ratio) where avg_reduction_ratio is
        the average ratio of consecutive remainders (lower = faster).
    """
    rems = np.asarray(remainders, dtype=np.float64).ravel()

    if len(rems) < 2:
        return len(rems), 0.0

    ratios = []
    for i in range(1, len(rems)):
        if rems[i - 1] > 0:
            ratios.append(rems[i] / rems[i - 1])

    if len(ratios) == 0:
        return len(rems), 0.0

    avg_ratio = float(np.mean(ratios))
    return len(rems), avg_ratio


# ---------------------------------------------------------------------------
# Primality witness check
# ---------------------------------------------------------------------------


def check_small_prime_divisors(
    n: int,
    n_primes: int = 100,
) -> tuple[bool, int]:
    """Check divisibility by small primes as a quick compositeness test.

    Before running expensive primality tests (Miller-Rabin), checking
    small prime divisors eliminates most composites cheaply.

    Args:
        n: the number to test.
        n_primes: how many small primes to check (default 100).

    Returns:
        (has_small_factor, smallest_factor) where has_small_factor is
        True if a factor was found, smallest_factor is the factor (0 if none).
    """
    val = int(n)
    if val < 2:
        return True, val

    # Generate small primes via simple sieve
    limit = max(10, n_primes * 10)
    sieve = np.ones(limit, dtype=bool)
    sieve[0] = sieve[1] = False
    for i in range(2, int(np.sqrt(limit)) + 1):
        if sieve[i]:
            sieve[i * i::i] = False
    primes = np.where(sieve)[0][:n_primes]

    for p in primes:
        p = int(p)
        if p >= val:
            break
        if val % p == 0:
            return True, p

    return False, 0


# ---------------------------------------------------------------------------
# Modular arithmetic overflow detection
# ---------------------------------------------------------------------------


def detect_modular_overflow(
    base: int,
    exponent: int,
    modulus: int,
) -> tuple[bool, int]:
    """Detect whether modular exponentiation risks intermediate overflow.

    If base * modulus exceeds int64 range, the intermediate
    multiplication in modular exponentiation will overflow without
    arbitrary-precision arithmetic.

    Args:
        base: base value.
        exponent: exponent value.
        modulus: modulus value.

    Returns:
        (would_overflow, safe_bits_needed) where would_overflow is True
        if base * modulus exceeds int64 range.
    """
    b = abs(int(base))
    m = abs(int(modulus))

    max_int64 = np.iinfo(np.int64).max

    # Check if b * m would overflow int64
    if m > 0 and b > max_int64 // m:
        bits_needed = int(np.ceil(np.log2(max(b, 1)) + np.log2(max(m, 1))))
        return True, bits_needed

    bits_needed = int(np.ceil(np.log2(max(b * m, 1)))) if b > 0 and m > 0 else 0
    return False, bits_needed

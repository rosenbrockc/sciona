"""Registry for number theory primitives and expansion atoms."""

from __future__ import annotations

NUMBER_THEORY_DECLARATIONS = {
    "validate_input_range": (
        "sciona.expansion_atoms.runtime_number_theory.validate_input_range",
        "ndarray, int -> tuple[int, bool]",
        "Check whether input values are within safe computation range.",
    ),
    "monitor_gcd_convergence": (
        "sciona.expansion_atoms.runtime_number_theory.monitor_gcd_convergence",
        "ndarray -> tuple[int, float]",
        "Monitor convergence rate of the Euclidean algorithm.",
    ),
    "check_small_prime_divisors": (
        "sciona.expansion_atoms.runtime_number_theory.check_small_prime_divisors",
        "int, int -> tuple[bool, int]",
        "Check divisibility by small primes as a quick compositeness test.",
    ),
    "detect_modular_overflow": (
        "sciona.expansion_atoms.runtime_number_theory.detect_modular_overflow",
        "int, int, int -> tuple[bool, int]",
        "Detect whether modular exponentiation risks intermediate overflow.",
    ),
}

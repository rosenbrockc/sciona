"""Fuzzing strategies for atom quality assurance.

Provides input generation and validation for property-based testing,
boundary value analysis, and behavioral equivalence detection.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Sequence

from ageom.ecosystem.models import BehavioralEquivalenceFlag, FuzzResult


# ---------------------------------------------------------------------------
# Input generation strategies
# ---------------------------------------------------------------------------


def generate_boundary_inputs(type_desc: str, count: int = 50) -> list[Any]:
    """Generate boundary-value inputs for a given type description.

    Supports: ``"float"``, ``"int"``, ``"np.ndarray"``, ``"list[float]"``,
    ``"str"``, ``"bool"``.
    """
    inputs: list[Any] = []

    if type_desc in ("float", "np.float64", "np.float32"):
        inputs.extend([
            0.0, -0.0, 1.0, -1.0,
            float("inf"), float("-inf"),
            1e-300, -1e-300,
            1e300, -1e300,
            float("nan"),
        ])
    elif type_desc in ("int", "np.int64", "np.int32"):
        inputs.extend([0, 1, -1, 2**31 - 1, -(2**31), 2**63 - 1])
    elif type_desc.startswith("np.ndarray") or type_desc.startswith("ndarray"):
        try:
            import numpy as np
            inputs.extend([
                np.array([]),
                np.array([0.0]),
                np.array([1.0, -1.0]),
                np.zeros(100),
                np.ones(100),
                np.full(10, float("inf")),
                np.full(10, float("nan")),
                np.arange(1000, dtype=np.float64),
            ])
        except ImportError:
            pass
    elif type_desc.startswith("list"):
        inputs.extend([[], [0.0], [1.0, -1.0], list(range(100))])
    elif type_desc == "str":
        inputs.extend(["", "a", "a" * 10000, "\x00", "\n\r\t"])
    elif type_desc == "bool":
        inputs.extend([True, False])

    return inputs[:count]


def generate_random_inputs(
    type_desc: str,
    count: int = 1000,
    seed: int = 42,
) -> list[Any]:
    """Generate random typed inputs for property-based testing."""
    try:
        import numpy as np
        rng = np.random.default_rng(seed)
    except ImportError:
        return []

    inputs: list[Any] = []

    if type_desc in ("float", "np.float64"):
        inputs = [float(rng.standard_normal()) for _ in range(count)]
    elif type_desc in ("int", "np.int64"):
        inputs = [int(rng.integers(-1000, 1000)) for _ in range(count)]
    elif type_desc.startswith("np.ndarray"):
        inputs = [rng.standard_normal(rng.integers(1, 200)) for _ in range(count)]
    elif type_desc.startswith("list"):
        inputs = [rng.standard_normal(rng.integers(1, 100)).tolist() for _ in range(count)]
    elif type_desc == "bool":
        inputs = [bool(rng.integers(0, 2)) for _ in range(count)]

    return inputs


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_output(output: Any, expected_type: str) -> list[str]:
    """Validate a function output against expected type.

    Returns a list of failure reasons (empty if valid).
    """
    reasons: list[str] = []

    if output is None:
        reasons.append("Output is None")
        return reasons

    # Check for NaN/Inf in numeric outputs
    if isinstance(output, float):
        if math.isnan(output):
            reasons.append("Output is NaN")
        if math.isinf(output):
            reasons.append("Output is Inf")

    try:
        import numpy as np
        if isinstance(output, np.ndarray):
            if np.any(np.isnan(output)):
                reasons.append("Output contains NaN values")
            if np.any(np.isinf(output)):
                reasons.append("Output contains Inf values")
    except ImportError:
        pass

    return reasons


# ---------------------------------------------------------------------------
# Behavioral equivalence
# ---------------------------------------------------------------------------


def check_behavioral_equivalence(
    func_a: Callable,
    func_b: Callable,
    inputs: Sequence[Any],
    *,
    tolerance: float = 1e-10,
    threshold: float = 0.95,
) -> BehavioralEquivalenceFlag | None:
    """Check if two functions produce equivalent outputs.

    Returns a flag if the match ratio exceeds the threshold.
    """
    matches = 0
    tested = 0

    for inp in inputs:
        try:
            out_a = func_a(inp)
            out_b = func_b(inp)
            tested += 1

            if _outputs_match(out_a, out_b, tolerance):
                matches += 1
        except Exception:
            continue

    if tested == 0:
        return None

    ratio = matches / tested

    if ratio >= threshold:
        return BehavioralEquivalenceFlag(
            atom_a_fqdn="",
            atom_a_hash="",
            atom_b_fqdn="",
            atom_b_hash="",
            match_ratio=ratio,
            sample_size=tested,
        )

    return None


def _outputs_match(a: Any, b: Any, tolerance: float) -> bool:
    """Check if two outputs are equivalent within tolerance."""
    if a is None and b is None:
        return True
    if a is None or b is None:
        return False

    try:
        import numpy as np
        if isinstance(a, np.ndarray) and isinstance(b, np.ndarray):
            if a.shape != b.shape:
                return False
            return bool(np.allclose(a, b, atol=tolerance, equal_nan=True))
    except ImportError:
        pass

    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if a == b:
            return True
        return abs(a - b) <= tolerance

    return a == b

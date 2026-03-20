"""Tests for fuzzing strategies — input generation, validation, behavioral equivalence."""

from __future__ import annotations

import math

import pytest

from sciona.ecosystem.fuzzing import (
    check_behavioral_equivalence,
    generate_boundary_inputs,
    generate_random_inputs,
    validate_output,
)


class TestBoundaryInputGeneration:
    def test_float_inputs(self):
        inputs = generate_boundary_inputs("float")
        assert len(inputs) > 0
        assert 0.0 in inputs
        assert any(math.isinf(x) for x in inputs if isinstance(x, float))

    def test_int_inputs(self):
        inputs = generate_boundary_inputs("int")
        assert 0 in inputs
        assert 1 in inputs
        assert -1 in inputs

    def test_ndarray_inputs(self):
        np = pytest.importorskip("numpy")
        inputs = generate_boundary_inputs("np.ndarray")
        assert len(inputs) > 0
        assert any(len(x) == 0 for x in inputs)

    def test_list_inputs(self):
        inputs = generate_boundary_inputs("list[float]")
        assert [] in inputs

    def test_string_inputs(self):
        inputs = generate_boundary_inputs("str")
        assert "" in inputs

    def test_bool_inputs(self):
        inputs = generate_boundary_inputs("bool")
        assert True in inputs
        assert False in inputs

    def test_count_limit(self):
        inputs = generate_boundary_inputs("float", count=3)
        assert len(inputs) <= 3

    def test_unknown_type(self):
        inputs = generate_boundary_inputs("unknown_type_xyz")
        assert inputs == []


class TestRandomInputGeneration:
    def test_float_inputs(self):
        np = pytest.importorskip("numpy")
        inputs = generate_random_inputs("float", count=100)
        assert len(inputs) == 100
        assert all(isinstance(x, float) for x in inputs)

    def test_int_inputs(self):
        np = pytest.importorskip("numpy")
        inputs = generate_random_inputs("int", count=50)
        assert len(inputs) == 50
        assert all(isinstance(x, int) for x in inputs)

    def test_deterministic_seed(self):
        np = pytest.importorskip("numpy")
        a = generate_random_inputs("float", count=10, seed=42)
        b = generate_random_inputs("float", count=10, seed=42)
        assert a == b

    def test_different_seeds(self):
        np = pytest.importorskip("numpy")
        a = generate_random_inputs("float", count=10, seed=1)
        b = generate_random_inputs("float", count=10, seed=2)
        assert a != b


class TestOutputValidation:
    def test_none_output(self):
        reasons = validate_output(None, "float")
        assert any("None" in r for r in reasons)

    def test_nan_output(self):
        reasons = validate_output(float("nan"), "float")
        assert any("NaN" in r for r in reasons)

    def test_inf_output(self):
        reasons = validate_output(float("inf"), "float")
        assert any("Inf" in r for r in reasons)

    def test_valid_float(self):
        reasons = validate_output(0.5, "float")
        assert reasons == []

    def test_ndarray_with_nan(self):
        np = pytest.importorskip("numpy")
        reasons = validate_output(np.array([1.0, float("nan")]), "np.ndarray")
        assert any("NaN" in r for r in reasons)

    def test_valid_ndarray(self):
        np = pytest.importorskip("numpy")
        reasons = validate_output(np.array([1.0, 2.0]), "np.ndarray")
        assert reasons == []


class TestBehavioralEquivalence:
    def test_identical_functions(self):
        def f(x): return x * 2
        def g(x): return x * 2
        inputs = list(range(100))
        result = check_behavioral_equivalence(f, g, inputs)
        assert result is not None
        assert result.match_ratio == 1.0

    def test_different_functions(self):
        def f(x): return x * 2
        def g(x): return x * 3
        inputs = list(range(1, 100))  # avoid 0 which matches
        result = check_behavioral_equivalence(f, g, inputs)
        assert result is None  # below 95% threshold

    def test_mostly_same(self):
        call_count = [0]
        def f(x): return x * 2
        def g(x):
            call_count[0] += 1
            if call_count[0] <= 2:
                return x * 3  # differ on first 2
            return x * 2

        inputs = list(range(100))
        result = check_behavioral_equivalence(f, g, inputs)
        assert result is not None  # 98% match

    def test_empty_inputs(self):
        def f(x): return x
        result = check_behavioral_equivalence(f, f, [])
        assert result is None

    def test_exception_handling(self):
        def f(x):
            if x == 5: raise ValueError
            return x
        def g(x): return x
        inputs = list(range(10))
        result = check_behavioral_equivalence(f, g, inputs)
        assert result is not None  # 9/9 non-error inputs match

    def test_custom_threshold(self):
        def f(x): return x
        def g(x): return x + (1 if x == 0 else 0)
        inputs = list(range(100))
        # 99% match: only x=0 differs
        result = check_behavioral_equivalence(f, g, inputs, threshold=0.99)
        assert result is not None

    def test_float_tolerance(self):
        def f(x): return x * 1.0
        def g(x): return x * 1.0 + 1e-15
        inputs = [float(i) for i in range(100)]
        result = check_behavioral_equivalence(f, g, inputs, tolerance=1e-10)
        assert result is not None

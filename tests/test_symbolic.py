"""Tests for sciona.ghost.symbolic – SymPy expression storage and compilation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from sciona.ghost.dimensions import (
    DIMENSIONLESS,
    HERTZ,
    JOULE,
    KELVIN,
    METER,
    MOLE,
    PASCAL,
    SECOND,
    VOLUME,
    WATT,
    DimensionalSignature,
)
from sciona.ghost.symbolic import (
    DimensionalError,
    SymbolicExpression,
    _infer_dim,
    deserialize_expr,
    serialize_expr,
)


# ---------------------------------------------------------------------------
# Serialisation roundtrip
# ---------------------------------------------------------------------------


class TestSerialization:
    def test_simple_expression(self):
        x, y = sp.symbols("x y")
        expr = x**2 + 2 * y
        s = serialize_expr(expr)
        restored = deserialize_expr(s)
        assert sp.simplify(expr - restored) == 0

    def test_equation(self):
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P * V, n * R * T)
        s = serialize_expr(eq)
        restored = deserialize_expr(s)
        assert isinstance(restored, sp.Eq)
        assert sp.simplify(restored.lhs - eq.lhs) == 0
        assert sp.simplify(restored.rhs - eq.rhs) == 0

    def test_transcendental(self):
        x = sp.Symbol("x")
        expr = sp.exp(-x) * sp.sin(x)
        s = serialize_expr(expr)
        restored = deserialize_expr(s)
        assert sp.simplify(expr - restored) == 0

    def test_derivative(self):
        x, t = sp.symbols("x t")
        expr = sp.Derivative(x**2, t)
        s = serialize_expr(expr)
        restored = deserialize_expr(s)
        assert restored == expr


# ---------------------------------------------------------------------------
# SymbolicExpression model
# ---------------------------------------------------------------------------


class TestSymbolicExpression:
    def test_to_sympy_roundtrip(self):
        x, y = sp.symbols("x y")
        expr = x * y + 1
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        restored = se.to_sympy()
        assert sp.simplify(expr - restored) == 0

    def test_to_numpy_lambda_basic(self):
        x, y = sp.symbols("x y")
        expr = x**2 + y
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        fn = se.to_numpy_lambda(["x", "y"])
        assert fn(3.0, 4.0) == pytest.approx(13.0)

    def test_to_numpy_lambda_with_constants(self):
        DM, freq = sp.symbols("DM freq")
        K = sp.Symbol("K")
        expr = K * DM * freq**(-2)
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"DM": "input", "freq": "input", "K": "constant"},
            constants={"K": 4.148808e3},
        )
        fn = se.to_numpy_lambda(["DM", "freq"])
        # K * 1.0 * (1000)^-2 = 4148.808 * 1e-6 = 0.004148808
        expected = 4.148808e3 * 1.0 * (1000.0 ** -2)
        assert fn(1.0, 1000.0) == pytest.approx(expected)

    def test_to_numpy_lambda_vectorized(self):
        x = sp.Symbol("x")
        expr = sp.sin(x)
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        fn = se.to_numpy_lambda(["x"])
        arr = np.linspace(0, np.pi, 100)
        result = fn(arr)
        np.testing.assert_allclose(result, np.sin(arr))

    def test_auto_input_vars(self):
        a, b = sp.symbols("a b")
        expr = a + b
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"a": "input", "b": "input"},
        )
        # Should auto-sort to ["a", "b"]
        fn = se.to_numpy_lambda()
        assert fn(1.0, 2.0) == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# Dimensional consistency checking
# ---------------------------------------------------------------------------


class TestDimensionalConsistency:
    def test_consistent_multiplication(self):
        """Force = mass * acceleration is dimensionally consistent."""
        m, a = sp.symbols("m a")
        expr = m * a
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"m": "input", "a": "input"},
            dim_map={
                "m": DimensionalSignature(M=1),          # kg
                "a": DimensionalSignature(L=1, T=-2),    # m/s²
            },
        )
        errors = se.check_dimensional_consistency()
        assert errors == []

    def test_inconsistent_addition(self):
        """Cannot add meters to seconds."""
        x, t = sp.symbols("x t")
        expr = x + t
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "t": "input"},
            dim_map={"x": METER, "t": SECOND},
        )
        errors = se.check_dimensional_consistency()
        assert len(errors) == 1
        assert "incompatible" in errors[0].lower() or "Cannot add" in errors[0]

    def test_consistent_equation(self):
        """P*V = n*R*T is dimensionally consistent with correct dim_map."""
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P * V, n * R * T)
        # P*V = Pa * m³ = J
        # n*R*T = mol * (J/(mol*K)) * K = J
        R_dim = JOULE.divide(MOLE).divide(KELVIN)
        se = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"P": "input", "V": "input", "n": "input", "R": "constant", "T": "input"},
            dim_map={
                "P": PASCAL,
                "V": DimensionalSignature(L=3),  # m³
                "n": MOLE,
                "R": R_dim,
                "T": KELVIN,
            },
        )
        errors = se.check_dimensional_consistency()
        assert errors == []

    def test_inconsistent_equation(self):
        """P = n*R*T is inconsistent when P has wrong dimension."""
        P, n, R, T = sp.symbols("P n R T")
        eq = sp.Eq(P, n * R * T)
        R_dim = JOULE.divide(MOLE).divide(KELVIN)
        se = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"P": "input", "n": "input", "R": "constant", "T": "input"},
            dim_map={
                "P": METER,  # Wrong! Should be PASCAL
                "n": MOLE,
                "R": R_dim,
                "T": KELVIN,
            },
        )
        errors = se.check_dimensional_consistency()
        assert len(errors) == 1

    def test_transcendental_requires_dimensionless(self):
        """exp(x) requires x to be dimensionless."""
        x = sp.Symbol("x")
        expr = sp.exp(x)
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
            dim_map={"x": METER},
        )
        errors = se.check_dimensional_consistency()
        assert len(errors) == 1
        assert "dimensionless" in errors[0].lower()

    def test_transcendental_with_dimensionless_ok(self):
        """exp(x) is fine when x is dimensionless."""
        x = sp.Symbol("x")
        expr = sp.exp(x)
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
            dim_map={"x": DIMENSIONLESS},
        )
        errors = se.check_dimensional_consistency()
        assert errors == []

    def test_no_dim_map_skips_check(self):
        """Empty dim_map should produce no errors."""
        x = sp.Symbol("x")
        expr = x + 1
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        errors = se.check_dimensional_consistency()
        assert errors == []

    def test_power_with_numeric_exponent(self):
        """x^2 preserves dimension correctly."""
        x = sp.Symbol("x")
        expr = x**2
        dim = _infer_dim(expr, {"x": METER}, sp)
        assert dim == DimensionalSignature(L=2)

    def test_derivative_dimension(self):
        """d(position)/d(time) = velocity."""
        x, t = sp.symbols("x t")
        expr = sp.Derivative(x, t)
        dim = _infer_dim(expr, {"x": METER, "t": SECOND}, sp)
        assert dim == DimensionalSignature(L=1, T=-1)  # velocity


# ---------------------------------------------------------------------------
# No SymPy at runtime
# ---------------------------------------------------------------------------


class TestNoSympyRuntime:
    def test_lambdified_function_is_numpy(self):
        """The callable returned by to_numpy_lambda should not import sympy."""
        x = sp.Symbol("x")
        expr = x**2 + 1
        se = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        fn = se.to_numpy_lambda(["x"])

        # The function should work with pure numpy input
        result = fn(np.array([1.0, 2.0, 3.0]))
        np.testing.assert_allclose(result, np.array([2.0, 5.0, 10.0]))

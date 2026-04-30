"""Tests for sciona.synthesizer.numpy_codegen – AOT SymPy-to-NumPy compilation."""

from __future__ import annotations

import math

import numpy as np
import pytest
import sympy as sp

from sciona.ghost.dimensions import (
    DIMENSIONLESS,
    KELVIN,
    JOULE,
    METER,
    MOLE,
    PASCAL,
    SECOND,
    DimensionalSignature,
)
from sciona.ghost.symbolic import SymbolicExpression, serialize_expr
from sciona.synthesizer.numpy_codegen import (
    sympy_to_numpy_source,
    sympy_to_numpy_source_multi,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _exec_source(source: str, func_name: str):
    """Compile and extract the named function from generated source."""
    ns = {"numpy": np, "np": np, "icontract": pytest.importorskip("icontract")}
    exec(source, ns)  # noqa: S102
    return ns[func_name]


# ---------------------------------------------------------------------------
# sympy_to_numpy_source
# ---------------------------------------------------------------------------


class TestSympyToNumpySource:
    def test_simple_polynomial(self):
        x, y = sp.symbols("x y")
        expr = x**2 + 2 * y
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        source = sympy_to_numpy_source(sym, "poly", input_vars=["x", "y"])
        assert "import sympy" not in source
        assert "def poly(x, y):" in source

        fn = _exec_source(source, "poly")
        assert fn(3.0, 4.0) == pytest.approx(17.0)

    def test_with_constants_inlined(self):
        DM, freq, K = sp.symbols("DM freq K")
        expr = K * DM * freq**(-2)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"DM": "input", "freq": "input", "K": "constant"},
            constants={"K": 4.148808e3},
        )
        source = sympy_to_numpy_source(sym, "delay", input_vars=["DM", "freq"])
        assert "import sympy" not in source
        assert "4148.808" in source or "4.148808" in source  # constant is inlined
        assert "def delay(DM, freq):" in source

        fn = _exec_source(source, "delay")
        expected = 4.148808e3 * 1.0 * (1000.0**-2)
        assert fn(1.0, 1000.0) == pytest.approx(expected)

    def test_transcendental(self):
        x = sp.Symbol("x")
        expr = sp.exp(-x) * sp.sin(x)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        source = sympy_to_numpy_source(sym, "damped_sin")
        fn = _exec_source(source, "damped_sin")

        arr = np.linspace(0, 5, 50)
        np.testing.assert_allclose(fn(arr), np.exp(-arr) * np.sin(arr))

    def test_validity_bounds_decorator(self):
        x = sp.Symbol("x")
        expr = sp.sqrt(x)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
            validity_bounds={"x": (0.0, None)},
        )
        source = sympy_to_numpy_source(sym, "safe_sqrt")
        assert "@icontract.require" in source
        assert ">= 0.0" in source

    def test_no_validity_when_disabled(self):
        x = sp.Symbol("x")
        expr = x**2
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
            validity_bounds={"x": (0.0, 100.0)},
        )
        source = sympy_to_numpy_source(sym, "sq", add_validity_checks=False)
        assert "@icontract" not in source

    def test_equation_uses_rhs(self):
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P, n * R * T / V)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"n": "input", "R": "constant", "T": "input", "V": "input", "P": "output"},
            constants={"R": 8.314},
        )
        source = sympy_to_numpy_source(sym, "calc_pressure", input_vars=["n", "T", "V"])
        fn = _exec_source(source, "calc_pressure")
        # PV = nRT => P = nRT/V
        expected = 1.0 * 8.314 * 300.0 / 0.001
        assert fn(1.0, 300.0, 0.001) == pytest.approx(expected)

    def test_vectorized_output(self):
        x = sp.Symbol("x")
        expr = x**3 - x
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        source = sympy_to_numpy_source(sym, "cubic")
        fn = _exec_source(source, "cubic")
        arr = np.array([0.0, 1.0, 2.0, 3.0])
        np.testing.assert_allclose(fn(arr), arr**3 - arr)


# ---------------------------------------------------------------------------
# sympy_to_numpy_source_multi (omnidirectional solving)
# ---------------------------------------------------------------------------


class TestOmnidirectionalSolving:
    def test_solve_for_volume(self):
        """PV = nRT solved for V => V = nRT/P."""
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P * V, n * R * T)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"P": "input", "V": "input", "n": "input", "R": "constant", "T": "input"},
            constants={"R": 8.314},
        )
        source = sympy_to_numpy_source_multi(
            sym, "calc_volume", solve_for="V", input_vars=["P", "n", "T"],
        )
        assert "def calc_volume(P, n, T):" in source

        fn = _exec_source(source, "calc_volume")
        # V = nRT/P = 1 * 8.314 * 300 / 101325
        expected = 1.0 * 8.314 * 300.0 / 101325.0
        assert fn(101325.0, 1.0, 300.0) == pytest.approx(expected)

    def test_solve_for_temperature(self):
        """PV = nRT solved for T => T = PV/(nR)."""
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P * V, n * R * T)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"P": "input", "V": "input", "n": "input", "R": "constant", "T": "input"},
            constants={"R": 8.314},
        )
        source = sympy_to_numpy_source_multi(
            sym, "calc_temp", solve_for="T", input_vars=["P", "V", "n"],
        )
        fn = _exec_source(source, "calc_temp")
        expected = 101325.0 * 0.0224 / (1.0 * 8.314)
        assert fn(101325.0, 0.0224, 1.0) == pytest.approx(expected)

    def test_non_equation_raises(self):
        x = sp.Symbol("x")
        expr = x**2
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        with pytest.raises(ValueError, match="Eq expression"):
            sympy_to_numpy_source_multi(sym, "bad", solve_for="x")


# ---------------------------------------------------------------------------
# No sympy in output
# ---------------------------------------------------------------------------


class TestNoSympyInOutput:
    def test_no_sympy_import_in_generated_code(self):
        x, y = sp.symbols("x y")
        expr = sp.sin(x) * sp.exp(-y)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        source = sympy_to_numpy_source(sym, "f")
        # The function body must not import sympy; docstrings may mention it
        assert "import sympy" not in source
        # The actual computation should use numpy, not sympy
        assert "numpy." in source or "np." in source

"""Tests for sciona.synthesizer.mojo_codegen -- AOT SymPy-to-Mojo compilation."""

from __future__ import annotations

import pytest
import sympy as sp

from sciona.ghost.symbolic import SymbolicExpression, serialize_expr
from sciona.synthesizer.mojo_codegen import (
    sympy_to_mojo_source,
    sympy_to_mojo_source_multi,
)


# ---------------------------------------------------------------------------
# sympy_to_mojo_source
# ---------------------------------------------------------------------------


class TestSympyToMojoSource:
    def test_simple_polynomial(self):
        x, y = sp.symbols("x y")
        expr = x**2 + 2 * y
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        source = sympy_to_mojo_source(sym, "poly", input_vars=["x", "y"])
        assert "fn poly_mojo(x: Float64, y: Float64) -> Float64:" in source
        assert "def poly(x, y):" in source

    def test_with_constants_inlined(self):
        DM, freq, K = sp.symbols("DM freq K")
        expr = K * DM * freq**(-2)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"DM": "input", "freq": "input", "K": "constant"},
            constants={"K": 4.148808e3},
        )
        source = sympy_to_mojo_source(sym, "delay", input_vars=["DM", "freq"])
        assert "import sympy" not in source
        assert "4148.808" in source or "4.148808" in source
        assert "fn delay_mojo(" in source

    def test_transcendental(self):
        x = sp.Symbol("x")
        expr = sp.exp(-x) * sp.sin(x)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        source = sympy_to_mojo_source(sym, "damped_sin")
        assert "math.sin" in source
        assert "math.exp" in source

    def test_no_sympy_in_output(self):
        x, y = sp.symbols("x y")
        expr = sp.sin(x) * sp.exp(-y)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        source = sympy_to_mojo_source(sym, "f")
        assert "import sympy" not in source

    def test_no_numpy_in_output(self):
        x = sp.Symbol("x")
        expr = sp.sqrt(x) + sp.cos(x)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        source = sympy_to_mojo_source(sym, "g")
        assert "import numpy" not in source
        assert "np." not in source
        assert "numpy." not in source

    def test_python_wrapper_present(self):
        x = sp.Symbol("x")
        expr = x**3
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        source = sympy_to_mojo_source(sym, "cube")
        # Mojo fn
        assert "fn cube_mojo(x: Float64) -> Float64:" in source
        # Python wrapper
        assert "def cube(x):" in source
        assert "cube_mojo(Float64(x))" in source

    def test_validity_bounds_emit_debug_assert(self):
        x = sp.Symbol("x")
        expr = sp.sqrt(x)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
            validity_bounds={"x": (0.0, None)},
        )
        source = sympy_to_mojo_source(sym, "safe_sqrt")
        assert "debug_assert" in source
        assert ">= 0.0" in source

    def test_no_validity_when_disabled(self):
        x = sp.Symbol("x")
        expr = x**2
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
            validity_bounds={"x": (0.0, 100.0)},
        )
        source = sympy_to_mojo_source(sym, "sq", add_validity_checks=False)
        assert "debug_assert" not in source

    def test_equation_uses_rhs(self):
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P, n * R * T / V)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"n": "input", "R": "constant", "T": "input", "V": "input", "P": "output"},
            constants={"R": 8.314},
        )
        source = sympy_to_mojo_source(sym, "calc_pressure", input_vars=["n", "T", "V"])
        assert "fn calc_pressure_mojo(" in source
        assert "def calc_pressure(" in source


# ---------------------------------------------------------------------------
# sympy_to_mojo_source_multi (omnidirectional solving)
# ---------------------------------------------------------------------------


class TestMojoOmnidirectionalSolving:
    def test_solve_for_volume(self):
        """PV = nRT solved for V => V = nRT/P."""
        P, V, n, R, T = sp.symbols("P V n R T")
        eq = sp.Eq(P * V, n * R * T)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(eq),
            variables={"P": "input", "V": "input", "n": "input", "R": "constant", "T": "input"},
            constants={"R": 8.314},
        )
        source = sympy_to_mojo_source_multi(
            sym, "calc_volume", solve_for="V", input_vars=["P", "n", "T"],
        )
        assert "fn calc_volume_mojo(P: Float64, n: Float64, T: Float64) -> Float64:" in source
        assert "def calc_volume(P, n, T):" in source

    def test_non_equation_raises(self):
        x = sp.Symbol("x")
        expr = x**2
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        with pytest.raises(ValueError, match="Eq expression"):
            sympy_to_mojo_source_multi(sym, "bad", solve_for="x")


# ---------------------------------------------------------------------------
# No external runtime deps in output
# ---------------------------------------------------------------------------


class TestNoExternalDeps:
    def test_no_sympy_import_in_generated_code(self):
        x, y = sp.symbols("x y")
        expr = sp.sin(x) * sp.exp(-y)
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input", "y": "input"},
        )
        source = sympy_to_mojo_source(sym, "f")
        assert "import sympy" not in source

    def test_no_numpy_import_in_generated_code(self):
        x = sp.Symbol("x")
        expr = sp.cos(x) + x**2
        sym = SymbolicExpression(
            srepr_str=serialize_expr(expr),
            variables={"x": "input"},
        )
        source = sympy_to_mojo_source(sym, "h")
        assert "import numpy" not in source
        assert "numpy." not in source

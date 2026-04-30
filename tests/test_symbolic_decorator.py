"""Tests for sciona.ghost.decorators – @symbolic_atom registration."""

from __future__ import annotations

import pytest
import sympy as sp

from sciona.ghost.abstract import AbstractScalar
from sciona.ghost.decorators import symbolic_atom
from sciona.ghost.dimensions import (
    DIMENSIONLESS,
    JOULE,
    KELVIN,
    METER,
    MOLE,
    PASCAL,
    SECOND,
    DimensionalSignature,
)
from sciona.ghost.registry import REGISTRY
from sciona.ghost.symbolic import SymbolicExpression


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _cleanup_registry(*names: str):
    """Remove test atoms from the global registry after a test."""
    for n in names:
        REGISTRY.pop(n, None)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


class TestSymbolicAtomRegistration:
    def test_basic_registration(self):
        x, y = sp.symbols("x y")

        def witness_add(a: AbstractScalar, b: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64")

        @symbolic_atom(
            witness=witness_add,
            expr=x + y,
            dim_map={"x": DIMENSIONLESS, "y": DIMENSIONLESS},
            name="_test_sym_add",
        )
        def sym_add(a: float, b: float) -> float:
            return a + b

        try:
            assert "_test_sym_add" in REGISTRY
            entry = REGISTRY["_test_sym_add"]
            assert entry["impl"] is sym_add
            assert entry["witness"] is witness_add
            assert entry["dim_signature"] == {"x": DIMENSIONLESS, "y": DIMENSIONLESS}
            assert isinstance(entry["symbolic"], SymbolicExpression)
            assert entry["symbolic"].srepr_str
        finally:
            _cleanup_registry("_test_sym_add")

    def test_registration_with_constants(self):
        DM, freq, K = sp.symbols("DM freq K")

        def witness_delay(dm: AbstractScalar, f: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64", min_val=0.0)

        @symbolic_atom(
            witness=witness_delay,
            expr=K * DM * freq**(-2),
            dim_map={
                "DM": DIMENSIONLESS,  # simplified
                "freq": DIMENSIONLESS,
                "K": DIMENSIONLESS,
            },
            constants={"K": 4.148808e3},
            name="_test_delay",
            bibliography=["Q12345"],
        )
        def delay_from_dm(DM: float, freq: float) -> float:
            return 4.148808e3 * DM * freq**(-2)

        try:
            entry = REGISTRY["_test_delay"]
            sym = entry["symbolic"]
            assert sym.constants == {"K": 4.148808e3}
            assert sym.bibliography == ["Q12345"]
            assert sym.variables["K"] == "constant"
            assert sym.variables["DM"] == "input"
        finally:
            _cleanup_registry("_test_delay")

    def test_registration_with_validity_bounds(self):
        x = sp.Symbol("x")

        def witness_sqrt(a: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64", min_val=0.0)

        @symbolic_atom(
            witness=witness_sqrt,
            expr=sp.sqrt(x),
            dim_map={"x": DIMENSIONLESS},
            validity_bounds={"x": (0.0, None)},
            name="_test_sqrt",
        )
        def safe_sqrt(x: float) -> float:
            return x**0.5

        try:
            entry = REGISTRY["_test_sqrt"]
            assert entry["symbolic"].validity_bounds == {"x": (0.0, None)}
        finally:
            _cleanup_registry("_test_sqrt")


# ---------------------------------------------------------------------------
# Import-time dimensional check
# ---------------------------------------------------------------------------


class TestImportTimeDimCheck:
    def test_consistent_passes(self):
        """Dimensionally consistent expression should register without error."""
        m, a = sp.symbols("m a")

        def witness_force(mass: AbstractScalar, accel: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64")

        @symbolic_atom(
            witness=witness_force,
            expr=m * a,
            dim_map={
                "m": DimensionalSignature(M=1),
                "a": DimensionalSignature(L=1, T=-2),
            },
            name="_test_force",
        )
        def calc_force(mass: float, accel: float) -> float:
            return mass * accel

        try:
            assert "_test_force" in REGISTRY
        finally:
            _cleanup_registry("_test_force")

    def test_inconsistent_raises(self):
        """Dimensionally inconsistent expression should raise ValueError at decoration time."""
        x, t = sp.symbols("x t")

        def witness_bad(a: AbstractScalar, b: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64")

        with pytest.raises(ValueError, match="Dimensional inconsistency"):
            @symbolic_atom(
                witness=witness_bad,
                expr=x + t,  # adding meters to seconds
                dim_map={"x": METER, "t": SECOND},
                name="_test_bad",
            )
            def bad_atom(x: float, t: float) -> float:
                return x + t

        # Should NOT be in registry
        assert "_test_bad" not in REGISTRY

    def test_skip_dim_check(self):
        """skip_dim_check=True should allow inconsistent expressions."""
        x, t = sp.symbols("x t")

        def witness_skip(a: AbstractScalar, b: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64")

        @symbolic_atom(
            witness=witness_skip,
            expr=x + t,
            dim_map={"x": METER, "t": SECOND},
            name="_test_skip",
            skip_dim_check=True,
        )
        def skip_atom(x: float, t: float) -> float:
            return x + t

        try:
            assert "_test_skip" in REGISTRY
        finally:
            _cleanup_registry("_test_skip")


# ---------------------------------------------------------------------------
# Callable still works
# ---------------------------------------------------------------------------


class TestCallablePreserved:
    def test_heavy_func_still_callable(self):
        """The decorated function should still be callable as normal."""
        x, y = sp.symbols("x y")

        def witness_mul(a: AbstractScalar, b: AbstractScalar) -> AbstractScalar:
            return AbstractScalar(dtype="float64")

        @symbolic_atom(
            witness=witness_mul,
            expr=x * y,
            dim_map={"x": DIMENSIONLESS, "y": DIMENSIONLESS},
            name="_test_mul",
        )
        def multiply(x: float, y: float) -> float:
            return x * y

        try:
            assert multiply(3.0, 4.0) == 12.0
        finally:
            _cleanup_registry("_test_mul")

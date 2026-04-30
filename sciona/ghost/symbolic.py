"""SymPy expression storage and compilation for symbolic atoms.

This module provides ``SymbolicExpression``, the internal representation
for physics equations stored as SymPy abstract syntax trees.  It supports:

* Lossless serialisation via ``sympy.srepr`` (AST, not LaTeX).
* AOT compilation to pure NumPy callables via ``sympy.lambdify``.
* Dimensional consistency checking against a ``DimensionalSignature`` map.

**SymPy is a build-time / synthesis-time dependency only.**  The callables
produced by ``to_numpy_lambda`` and the source strings produced by
``to_numpy_source`` have zero SymPy runtime imports.
"""

from __future__ import annotations

import logging
from fractions import Fraction
from typing import Any, Callable, Sequence

from pydantic import BaseModel, Field

from sciona.ghost.dimensions import (
    DIMENSIONLESS,
    UNKNOWN_DIMENSION,
    DimensionalSignature,
)

logger = logging.getLogger(__name__)

# Lazy import guard – SymPy may not be installed in minimal envs.
_sympy = None


def _ensure_sympy():
    global _sympy
    if _sympy is None:
        try:
            import sympy as _sp
            _sympy = _sp
        except ImportError as exc:
            raise ImportError(
                "SymPy >= 1.12 is required for symbolic atoms.  "
                "Install it with: pip install 'sciona[sympy]'"
            ) from exc
    return _sympy


# ---------------------------------------------------------------------------
# Core model
# ---------------------------------------------------------------------------


class SymbolicExpression(BaseModel):
    """A SymPy expression with physics metadata.

    The expression is stored as an ``srepr`` string for deterministic
    serialisation.  At registration time the live SymPy ``Expr`` is held
    transiently; for persistence the ``srepr_str`` is the source of truth.
    """

    srepr_str: str = Field(
        ..., description="sympy.srepr() serialisation of the expression"
    )
    variables: dict[str, str] = Field(
        default_factory=dict,
        description='Map variable name -> role: "input", "output", "parameter", "constant"',
    )
    dim_map: dict[str, DimensionalSignature] = Field(
        default_factory=dict,
        description="Map variable name -> DimensionalSignature",
    )
    validity_bounds: dict[str, tuple[float | None, float | None]] = Field(
        default_factory=dict,
        description="Map variable name -> (min, max) validity range",
    )
    constants: dict[str, float] = Field(
        default_factory=dict,
        description="Named physical constants with their numerical values",
    )
    bibliography: list[str] = Field(
        default_factory=list,
        description="Reference keys for audit graph provenance",
    )

    model_config = {"arbitrary_types_allowed": True}

    # ----- SymPy expression access -----

    def to_sympy(self) -> Any:
        """Reconstruct the live SymPy expression from ``srepr_str``."""
        return _eval_srepr(self.srepr_str)

    # ----- NumPy compilation -----

    def to_numpy_lambda(
        self,
        input_vars: Sequence[str] | None = None,
    ) -> Callable:
        """Compile to a pure NumPy callable (zero SymPy runtime dependency).

        Args:
            input_vars: Ordered variable names for the lambda's positional
                arguments.  If omitted, uses all variables with role
                ``"input"`` in sorted order.

        Returns:
            A callable ``f(*args) -> numpy result``.
        """
        sp = _ensure_sympy()
        expr = _eval_srepr(self.srepr_str)

        if input_vars is None:
            input_vars = sorted(
                name for name, role in self.variables.items()
                if role == "input"
            )

        symbols = [sp.Symbol(v) for v in input_vars]

        # Substitute named constants
        subs = {sp.Symbol(k): v for k, v in self.constants.items()}
        expr_substituted = expr.subs(subs) if subs else expr

        return sp.lambdify(symbols, expr_substituted, modules=["numpy"])

    # ----- Dimensional consistency -----

    def check_dimensional_consistency(self) -> list[str]:
        """Verify the expression is dimensionally consistent.

        Walks the SymPy expression tree and propagates dimensions through
        operations.  Returns a list of error messages (empty = consistent).
        """
        sp = _ensure_sympy()
        expr = _eval_srepr(self.srepr_str)
        errors: list[str] = []

        if not self.dim_map:
            return errors

        try:
            inferred = _infer_dim(expr, self.dim_map, sp)
            if inferred.is_unknown:
                errors.append(
                    "Cannot verify dimensional consistency: inferred unknown dimension"
                )
        except DimensionalError as e:
            errors.append(str(e))
        return errors


# ---------------------------------------------------------------------------
# srepr eval helper
# ---------------------------------------------------------------------------


def _eval_srepr(srepr_str: str) -> Any:
    """Evaluate an srepr string back into a live SymPy object."""
    sp = _ensure_sympy()
    # Build a namespace containing all SymPy names needed for eval
    ns: dict[str, Any] = {}
    for name in dir(sp):
        obj = getattr(sp, name)
        ns[name] = obj
    # Also include core classes
    ns.update({
        "Symbol": sp.Symbol,
        "Integer": sp.Integer,
        "Rational": sp.Rational,
        "Float": sp.Float,
        "Eq": sp.Eq,
        "Function": sp.Function,
        "Derivative": sp.Derivative,
        "Integral": sp.Integral,
        "Tuple": sp.Tuple,
        "Pow": sp.Pow,
        "Add": sp.Add,
        "Mul": sp.Mul,
    })
    return eval(srepr_str, {"__builtins__": {}}, ns)  # noqa: S307


# ---------------------------------------------------------------------------
# Serialisation helpers
# ---------------------------------------------------------------------------


def serialize_expr(expr: Any) -> str:
    """Serialize a SymPy expression to its ``srepr`` AST string."""
    sp = _ensure_sympy()
    return sp.srepr(expr)


def deserialize_expr(s: str) -> Any:
    """Deserialize an ``srepr`` string back into a SymPy expression."""
    return _eval_srepr(s)


# ---------------------------------------------------------------------------
# Dimensional inference engine
# ---------------------------------------------------------------------------


class DimensionalError(Exception):
    """Raised when a dimensional inconsistency is detected."""


def _infer_dim(
    expr: Any,
    dim_map: dict[str, DimensionalSignature],
    sp: Any,
) -> DimensionalSignature:
    """Recursively infer the dimensional signature of a SymPy expression.

    Raises ``DimensionalError`` on inconsistency.
    """
    # Symbol -> look up in dim_map
    if isinstance(expr, sp.Symbol):
        name = str(expr)
        if name in dim_map:
            dim = dim_map[name]
            if dim.is_unknown:
                raise DimensionalError(f"Unknown dimension for symbol '{name}'")
            return dim
        raise DimensionalError(f"Unknown dimension for symbol '{name}'")

    # Numeric literals are dimensionless
    from sympy.core.numbers import (
        Exp1, Half, NegativeOne, One, Pi, Zero,
    )
    if isinstance(expr, (sp.Integer, sp.Float, sp.Rational,
                          One, Zero, NegativeOne, Half, Pi, Exp1)):
        return DIMENSIONLESS
    # Number base class catch-all
    if expr.is_number:
        return DIMENSIONLESS

    # Addition / Subtraction: all terms must have the same dimension
    if isinstance(expr, sp.Add):
        dims = [_infer_dim(arg, dim_map, sp) for arg in expr.args]
        if any(dim.is_unknown for dim in dims):
            return UNKNOWN_DIMENSION
        base = dims[0]
        for i, d in enumerate(dims[1:], 1):
            if not base.is_compatible(d):
                raise DimensionalError(
                    f"Cannot add terms with incompatible dimensions: "
                    f"{base.to_compact()} + {d.to_compact()} "
                    f"(term {i} of Add)"
                )
        return base

    # Multiplication: multiply dimensions
    if isinstance(expr, sp.Mul):
        result = DIMENSIONLESS
        for arg in expr.args:
            result = result.multiply(_infer_dim(arg, dim_map, sp))
        return result

    # Power: base^exponent.  Exponent must be dimensionless and rational.
    if isinstance(expr, sp.Pow):
        base_expr, exp_expr = expr.args
        base_dim = _infer_dim(base_expr, dim_map, sp)
        exp_dim = _infer_dim(exp_expr, dim_map, sp)
        if not exp_dim.is_dimensionless:
            raise DimensionalError(
                f"Exponent must be dimensionless, got {exp_dim.to_compact()}"
            )
        # If exponent is a numeric constant, scale the base dimension
        if exp_expr.is_number:
            if base_dim.is_dimensionless:
                return DIMENSIONLESS
            n = _sympy_number_to_fraction(exp_expr)
            if n is None:
                raise DimensionalError(
                    "Exponent for a dimensioned quantity must be rational, "
                    f"got {exp_expr}"
                )
            return base_dim.power(n)
        # If exponent is symbolic but dimensionless, base must also be dimensionless
        if not base_dim.is_dimensionless:
            raise DimensionalError(
                f"Cannot raise dimensioned quantity ({base_dim.to_compact()}) "
                f"to a symbolic power"
            )
        return DIMENSIONLESS

    # Equality: both sides must have the same dimension
    if isinstance(expr, sp.Eq):
        lhs_dim = _infer_dim(expr.lhs, dim_map, sp)
        rhs_dim = _infer_dim(expr.rhs, dim_map, sp)
        if lhs_dim.is_unknown or rhs_dim.is_unknown:
            return UNKNOWN_DIMENSION
        if not lhs_dim.is_compatible(rhs_dim):
            raise DimensionalError(
                f"Equation sides have incompatible dimensions: "
                f"LHS={lhs_dim.to_compact()} vs RHS={rhs_dim.to_compact()}"
            )
        return lhs_dim

    # Transcendental functions (exp, log, sin, cos, etc.): argument must
    # be dimensionless, result is dimensionless.
    # SymPy represents these as Application subclasses; check by class name.
    _TRANSCENDENTAL_NAMES = frozenset({
        "exp", "log", "ln", "sin", "cos", "tan",
        "asin", "acos", "atan", "atan2",
        "sinh", "cosh", "tanh", "asinh", "acosh", "atanh",
    })
    expr_class_name = type(expr).__name__
    if expr_class_name in _TRANSCENDENTAL_NAMES:
        for arg in expr.args:
            arg_dim = _infer_dim(arg, dim_map, sp)
            if arg_dim.is_unknown:
                raise DimensionalError(
                    f"Argument to {expr_class_name} has unknown dimension"
                )
            if not arg_dim.is_dimensionless:
                raise DimensionalError(
                    f"Argument to {expr_class_name} must be "
                    f"dimensionless, got {arg_dim.to_compact()}"
                )
        return DIMENSIONLESS

    # Abs: preserves dimension
    if isinstance(expr, sp.Abs):
        return _infer_dim(expr.args[0], dim_map, sp)

    # Derivative: dim(f)/dim(x)
    if isinstance(expr, sp.Derivative):
        func_dim = _infer_dim(expr.args[0], dim_map, sp)
        for var, order in expr.variable_count:
            var_dim = _infer_dim(var, dim_map, sp)
            for _ in range(order):
                func_dim = func_dim.divide(var_dim)
        return func_dim

    # Fallback: if we can't determine, keep the uncertainty explicit.
    logger.debug("Cannot infer dimension for %s (%s), marking unknown",
                 expr, type(expr).__name__)
    return UNKNOWN_DIMENSION


def _sympy_number_to_fraction(expr: Any) -> Fraction | None:
    """Convert a rational SymPy numeric expression to an exact Fraction."""
    if getattr(expr, "is_Rational", False):
        return Fraction(int(expr.p), int(expr.q))
    if getattr(expr, "is_Integer", False):
        return Fraction(int(expr), 1)
    if getattr(expr, "is_Float", False):
        return Fraction(str(expr))
    return None


def _try_sqrt_dim(dim: DimensionalSignature) -> DimensionalSignature:
    """Compute sqrt of a dimensional signature using exact half powers."""
    return dim.power(Fraction(1, 2))

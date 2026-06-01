"""Ahead-of-Time (AOT) SymPy-to-Mojo code generation.

Converts ``SymbolicExpression`` objects into compiled Mojo ``fn``
definitions with a Python-callable ``def`` wrapper.  The generated code
has **zero** ``sympy`` or ``numpy`` imports — it uses Mojo's native
``math`` module for SIMD-aware transcendentals and ``Float64`` scalars.

Parallel to ``numpy_codegen.py``: SymPy is the IR, Mojo is the machine
code.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

logger = logging.getLogger(__name__)

# Lazy SymPy import (build-time only)
_sympy = None


def _ensure_sympy():
    global _sympy
    if _sympy is None:
        import sympy as _sp
        _sympy = _sp
    return _sympy


# ---------------------------------------------------------------------------
# Mojo-specific SymPy printer
# ---------------------------------------------------------------------------

_MATH_FUNCS = {
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2",
    "sinh", "cosh", "tanh",
    "exp", "log", "log2", "log10",
    "sqrt", "abs", "floor", "ceil",
}


class _MojoPrinter:
    """Minimal SymPy expression printer targeting Mojo's ``math`` module.

    This is intentionally *not* a full ``sympy.printing.CodePrinter``
    subclass — we only need scalar expression printing for AOT atoms,
    and keeping the dependency surface small avoids coupling to SymPy
    printer internals that change across versions.
    """

    def doprint(self, expr) -> str:
        sp = _ensure_sympy()
        return self._print(expr, sp)

    def _print(self, expr, sp) -> str:  # noqa: C901 — flat dispatch is clearer here
        if isinstance(expr, sp.Number):
            return self._print_number(expr, sp)
        if isinstance(expr, sp.Symbol):
            return str(expr)
        if isinstance(expr, sp.Add):
            terms = [self._print(a, sp) for a in expr.args]
            return " + ".join(terms)
        if isinstance(expr, sp.Mul):
            return self._print_mul(expr, sp)
        if isinstance(expr, sp.Pow):
            base = self._print(expr.args[0], sp)
            exp_ = self._print(expr.args[1], sp)
            # sqrt
            if expr.args[1] == sp.Rational(1, 2):
                return f"math.sqrt({base})"
            # reciprocal sqrt
            if expr.args[1] == sp.Rational(-1, 2):
                return f"(1.0 / math.sqrt({base}))"
            return f"({base}) ** ({exp_})"
        # Function calls (sin, exp, etc.)
        if isinstance(expr, sp.Function):
            func_name = type(expr).__name__
            if func_name == "Abs":
                func_name = "abs"
            args = ", ".join(self._print(a, sp) for a in expr.args)
            if func_name in _MATH_FUNCS:
                return f"math.{func_name}({args})"
            return f"{func_name}({args})"
        # Rational numbers
        if isinstance(expr, sp.Rational):
            return f"({float(expr)})"
        # Fallback: use str and hope for the best
        return str(expr)

    def _print_number(self, expr, sp) -> str:
        if isinstance(expr, sp.Integer):
            return f"Float64({int(expr)})"
        if isinstance(expr, sp.Rational):
            return f"({float(expr)})"
        if isinstance(expr, sp.Float):
            return str(float(expr))
        if expr is sp.pi:
            return "math.pi"
        if expr is sp.E:
            return "math.e"
        return str(float(expr))

    def _print_mul(self, expr, sp) -> str:
        # Handle negative coefficients cleanly
        coeff, rest = expr.as_coeff_Mul()
        if coeff == -1:
            return f"-({self._print(rest, sp)})"
        factors = [self._print(a, sp) for a in expr.args]
        return " * ".join(factors)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sympy_to_mojo_source(
    symbolic: Any,
    func_name: str,
    *,
    input_vars: Sequence[str] | None = None,
    docstring: str = "",
    add_validity_checks: bool = True,
) -> str:
    """Compile a ``SymbolicExpression`` to a Mojo ``fn`` + Python ``def`` wrapper.

    Args:
        symbolic: A ``SymbolicExpression`` instance.
        func_name: Name for the generated function.
        input_vars: Ordered variable names for function arguments.
            If omitted, uses all variables with role ``"input"`` sorted.
        docstring: Optional docstring to embed.
        add_validity_checks: If ``True``, add ``debug_assert`` guards
            from the expression's ``validity_bounds``.

    Returns:
        Source string containing a Mojo ``fn`` and a Python ``def`` wrapper.
        No ``sympy`` or ``numpy`` imports.
    """
    sp = _ensure_sympy()
    from sciona.ghost.symbolic import _eval_srepr

    expr = _eval_srepr(symbolic.srepr_str)

    if input_vars is None:
        input_vars = sorted(
            name for name, role in symbolic.variables.items()
            if role == "input"
        )

    # Substitute named constants
    subs = {}
    for const_name, const_val in symbolic.constants.items():
        subs[sp.Symbol(const_name)] = sp.Float(const_val)
    expr_substituted = expr.subs(subs) if subs else expr

    # For Eq objects, use the RHS
    if isinstance(expr_substituted, sp.Eq):
        code_expr = expr_substituted.rhs
    else:
        code_expr = expr_substituted

    # Generate Mojo code
    printer = _MojoPrinter()
    mojo_code = printer.doprint(code_expr)

    # Collect math functions used
    math_funcs_used = sorted(_MATH_FUNCS & set(mojo_code.replace("math.", " ").split()))

    lines: list[str] = []

    # Mojo imports
    lines.append("from math import " + ", ".join(math_funcs_used) if math_funcs_used else "import math")
    lines.append("")

    # Mojo fn signature
    mojo_params = ", ".join(f"{v}: Float64" for v in input_vars)
    mojo_fn_name = f"{func_name}_mojo"
    lines.append(f"fn {mojo_fn_name}({mojo_params}) -> Float64:")
    doc = docstring or f"AOT-compiled from SymPy: {symbolic.srepr_str[:60]}..."
    lines.append(f'    """{doc}"""')

    # Constant comments
    for const_name, const_val in symbolic.constants.items():
        lines.append(f"    # {const_name} = {const_val}")

    # Validity bounds as debug_assert
    if add_validity_checks and symbolic.validity_bounds:
        for var_name, (lo, hi) in symbolic.validity_bounds.items():
            if var_name not in input_vars:
                continue
            if lo is not None:
                lines.append(
                    f'    debug_assert({var_name} >= {float(lo)}, "{var_name} must be >= {lo}")'
                )
            if hi is not None:
                lines.append(
                    f'    debug_assert({var_name} <= {float(hi)}, "{var_name} must be <= {hi}")'
                )

    lines.append(f"    return {mojo_code}")
    lines.append("")

    # Python-callable wrapper
    py_params = ", ".join(input_vars)
    py_args = ", ".join(f"Float64({v})" for v in input_vars)
    lines.append(f"def {func_name}({py_params}):")
    lines.append(f'    """Python-callable wrapper for Mojo AOT function."""')
    lines.append(f"    return {mojo_fn_name}({py_args})")

    return "\n".join(lines)


def sympy_to_mojo_source_multi(
    symbolic: Any,
    func_name: str,
    solve_for: str,
    *,
    input_vars: Sequence[str] | None = None,
    docstring: str = "",
    add_validity_checks: bool = True,
) -> str:
    """Compile a ``SymbolicExpression`` equation solved for a specific variable.

    Omnidirectional solving variant — one SymPy equation generates
    different Mojo functions depending on which variable is the output.

    Args:
        symbolic: A ``SymbolicExpression`` with an ``Eq`` expression.
        func_name: Name for the generated function.
        solve_for: Variable name to solve for (becomes the output).
        input_vars: Remaining input variables.  If omitted, all variables
            except ``solve_for`` and constants are used.
        docstring: Optional docstring.
        add_validity_checks: Add debug_assert guards.

    Returns:
        Mojo function source string.
    """
    sp = _ensure_sympy()
    from sciona.ghost.symbolic import _eval_srepr

    expr = _eval_srepr(symbolic.srepr_str)

    if not isinstance(expr, sp.Eq):
        raise ValueError(
            f"sympy_to_mojo_source_multi requires an Eq expression, "
            f"got {type(expr).__name__}"
        )

    target_sym = sp.Symbol(solve_for)
    solutions = sp.solve(expr, target_sym)
    if not solutions:
        raise ValueError(f"Cannot solve expression for '{solve_for}'")

    solution = solutions[0]

    # Substitute constants
    subs = {sp.Symbol(k): sp.Float(v) for k, v in symbolic.constants.items()}
    solution = solution.subs(subs) if subs else solution

    if input_vars is None:
        const_names = set(symbolic.constants)
        input_vars = sorted(
            name for name, role in symbolic.variables.items()
            if role == "input" and name != solve_for and name not in const_names
        )

    printer = _MojoPrinter()
    mojo_code = printer.doprint(solution)

    math_funcs_used = sorted(_MATH_FUNCS & set(mojo_code.replace("math.", " ").split()))

    lines: list[str] = []

    lines.append("from math import " + ", ".join(math_funcs_used) if math_funcs_used else "import math")
    lines.append("")

    mojo_params = ", ".join(f"{v}: Float64" for v in input_vars)
    mojo_fn_name = f"{func_name}_mojo"
    lines.append(f"fn {mojo_fn_name}({mojo_params}) -> Float64:")
    doc = docstring or f"Solved for {solve_for} from: {symbolic.srepr_str[:50]}..."
    lines.append(f'    """{doc}"""')

    for const_name, const_val in symbolic.constants.items():
        lines.append(f"    # {const_name} = {const_val}")

    if add_validity_checks and symbolic.validity_bounds:
        for var_name, (lo, hi) in symbolic.validity_bounds.items():
            if var_name not in input_vars:
                continue
            if lo is not None:
                lines.append(
                    f'    debug_assert({var_name} >= {float(lo)}, "{var_name} must be >= {lo}")'
                )
            if hi is not None:
                lines.append(
                    f'    debug_assert({var_name} <= {float(hi)}, "{var_name} must be <= {hi}")'
                )

    lines.append(f"    return {mojo_code}")
    lines.append("")

    py_params = ", ".join(input_vars)
    py_args = ", ".join(f"Float64({v})" for v in input_vars)
    lines.append(f"def {func_name}({py_params}):")
    lines.append(f'    """Python-callable wrapper for Mojo AOT function."""')
    lines.append(f"    return {mojo_fn_name}({py_args})")

    return "\n".join(lines)

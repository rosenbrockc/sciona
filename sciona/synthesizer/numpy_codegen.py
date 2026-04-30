"""Ahead-of-Time (AOT) SymPy-to-NumPy code generation.

Converts ``SymbolicExpression`` objects into pure NumPy Python source
strings.  The generated code has **zero** ``sympy`` imports — it is
equivalent to a hand-written vectorised NumPy function.

This module is the "compiler backend" described in the symbolic_math
design doc: SymPy is the IR, NumPy is the machine code.
"""

from __future__ import annotations

import logging
import textwrap
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


def sympy_to_numpy_source(
    symbolic: Any,
    func_name: str,
    *,
    input_vars: Sequence[str] | None = None,
    docstring: str = "",
    add_validity_checks: bool = True,
) -> str:
    """Compile a ``SymbolicExpression`` to a pure NumPy function source string.

    Args:
        symbolic: A ``SymbolicExpression`` instance.
        func_name: Name for the generated Python function.
        input_vars: Ordered variable names for function arguments.
            If omitted, uses all variables with role ``"input"`` sorted.
        docstring: Optional docstring to embed.
        add_validity_checks: If ``True``, add ``@icontract.require``
            decorators from the expression's ``validity_bounds``.

    Returns:
        A complete Python function definition as a string, using only
        ``numpy`` and optionally ``icontract``.  No ``sympy`` imports.
    """
    sp = _ensure_sympy()
    from sympy.printing.numpy import NumPyPrinter

    from sciona.ghost.symbolic import _eval_srepr

    expr = _eval_srepr(symbolic.srepr_str)

    # Determine input variables
    if input_vars is None:
        input_vars = sorted(
            name for name, role in symbolic.variables.items()
            if role == "input"
        )

    # Substitute named constants into the expression
    subs = {}
    for const_name, const_val in symbolic.constants.items():
        subs[sp.Symbol(const_name)] = sp.Float(const_val)
    expr_substituted = expr.subs(subs) if subs else expr

    # For Eq objects, generate a function that returns the RHS solved
    # (or the full expression if not an Eq)
    if isinstance(expr_substituted, sp.Eq):
        # Default: return the RHS (assumes LHS is the output variable)
        code_expr = expr_substituted.rhs
    else:
        code_expr = expr_substituted

    # Generate NumPy code using SymPy's printer
    printer = NumPyPrinter()
    numpy_code = printer.doprint(code_expr)

    # Build function lines
    lines: list[str] = []

    # Validity bound decorators
    if add_validity_checks and symbolic.validity_bounds:
        for var_name, (lo, hi) in symbolic.validity_bounds.items():
            if var_name not in input_vars:
                continue
            if lo is not None and hi is not None:
                lines.append(
                    f"@icontract.require("
                    f"lambda {var_name}: {lo} <= {var_name} <= {hi}, "
                    f'"{var_name} must be in [{lo}, {hi}]")'
                )
            elif lo is not None:
                lines.append(
                    f"@icontract.require("
                    f"lambda {var_name}: {var_name} >= {lo}, "
                    f'"{var_name} must be >= {lo}")'
                )
            elif hi is not None:
                lines.append(
                    f"@icontract.require("
                    f"lambda {var_name}: {var_name} <= {hi}, "
                    f'"{var_name} must be <= {hi}")'
                )

    # Function signature
    params = ", ".join(input_vars)
    lines.append(f"def {func_name}({params}):")

    # Docstring
    doc = docstring or f"AOT-compiled from SymPy: {symbolic.srepr_str[:60]}..."
    lines.append(f'    """{doc}"""')

    # Constant comments (for auditability)
    for const_name, const_val in symbolic.constants.items():
        lines.append(f"    # {const_name} = {const_val}")

    # Body
    lines.append(f"    return {numpy_code}")

    return "\n".join(lines)


def sympy_to_numpy_source_multi(
    symbolic: Any,
    func_name: str,
    solve_for: str,
    *,
    input_vars: Sequence[str] | None = None,
    docstring: str = "",
    add_validity_checks: bool = True,
) -> str:
    """Compile a ``SymbolicExpression`` equation solved for a specific variable.

    This implements "omnidirectional solving" — one SymPy equation can
    generate different NumPy functions depending on which variable is the
    output.

    Args:
        symbolic: A ``SymbolicExpression`` with an ``Eq`` expression.
        func_name: Name for the generated function.
        solve_for: Variable name to solve for (becomes the output).
        input_vars: Remaining input variables.  If omitted, all variables
            except ``solve_for`` and constants are used.
        docstring: Optional docstring.
        add_validity_checks: Add validity bound decorators.

    Returns:
        Python function source string.
    """
    sp = _ensure_sympy()
    from sympy.printing.numpy import NumPyPrinter

    from sciona.ghost.symbolic import _eval_srepr

    expr = _eval_srepr(symbolic.srepr_str)

    if not isinstance(expr, sp.Eq):
        raise ValueError(
            f"sympy_to_numpy_source_multi requires an Eq expression, "
            f"got {type(expr).__name__}"
        )

    target_sym = sp.Symbol(solve_for)
    solutions = sp.solve(expr, target_sym)
    if not solutions:
        raise ValueError(f"Cannot solve expression for '{solve_for}'")

    # Take the first solution (for simple equations there's typically one)
    solution = solutions[0]

    # Substitute constants
    subs = {sp.Symbol(k): sp.Float(v) for k, v in symbolic.constants.items()}
    solution = solution.subs(subs) if subs else solution

    # Determine input vars (all except solve_for and constants)
    if input_vars is None:
        const_names = set(symbolic.constants)
        input_vars = sorted(
            name for name, role in symbolic.variables.items()
            if role == "input" and name != solve_for and name not in const_names
        )

    # Generate code
    printer = NumPyPrinter()
    numpy_code = printer.doprint(solution)

    lines: list[str] = []

    if add_validity_checks and symbolic.validity_bounds:
        for var_name, (lo, hi) in symbolic.validity_bounds.items():
            if var_name not in input_vars:
                continue
            if lo is not None:
                lines.append(
                    f"@icontract.require("
                    f"lambda {var_name}: {var_name} >= {lo}, "
                    f'"{var_name} must be >= {lo}")'
                )
            if hi is not None:
                lines.append(
                    f"@icontract.require("
                    f"lambda {var_name}: {var_name} <= {hi}, "
                    f'"{var_name} must be <= {hi}")'
                )

    params = ", ".join(input_vars)
    lines.append(f"def {func_name}({params}):")
    doc = docstring or f"Solved for {solve_for} from: {symbolic.srepr_str[:50]}..."
    lines.append(f'    """{doc}"""')
    for const_name, const_val in symbolic.constants.items():
        lines.append(f"    # {const_name} = {const_val}")
    lines.append(f"    return {numpy_code}")

    return "\n".join(lines)

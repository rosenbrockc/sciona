"""Convenience decorators for registering symbolic physics atoms.

``@symbolic_atom`` is the primary user-facing decorator for new physics
atoms.  It combines:

1. SymPy expression storage (via ``SymbolicExpression``).
2. Import-time dimensional consistency checking.
3. Standard ``@register_atom`` registration with dim_map and symbolic
   metadata in the registry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Sequence

from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.registry import register_atom
from sciona.ghost.symbolic import SymbolicExpression, serialize_expr

if TYPE_CHECKING:
    pass


def symbolic_atom(
    witness: Callable,
    *,
    expr: Any,
    dim_map: dict[str, DimensionalSignature],
    name: str | None = None,
    validity_bounds: dict[str, tuple[float | None, float | None]] | None = None,
    constants: dict[str, float] | None = None,
    bibliography: list[str] | None = None,
    variables: dict[str, str] | None = None,
    skip_dim_check: bool = False,
) -> Callable:
    """Decorator that registers a symbolic physics atom.

    This is the primary entry point for defining new physics atoms with
    full SymPy expression storage and dimensional analysis.

    Args:
        witness: Ghost witness callable (same as ``@register_atom``).
        expr: A SymPy expression (``Expr`` or ``Eq``).  Stored as an AST
            via ``sympy.srepr``, not LaTeX.
        dim_map: Maps each variable/parameter name to its
            ``DimensionalSignature``.  Used for import-time dimensional
            validation and compile-time edge checking.
        name: Optional explicit registry key.
        validity_bounds: Per-variable ``(min, max)`` validity ranges.
            ``None`` on either bound means unbounded.
        constants: Named physical constants with their numerical values.
            These are substituted into the expression before NumPy
            compilation.
        bibliography: Reference keys (e.g. Wikidata Q-items, DOIs) for
            the audit graph.
        variables: Maps variable names to roles (``"input"``,
            ``"output"``, ``"parameter"``, ``"constant"``).  If omitted,
            all symbols in ``dim_map`` that are not in ``constants`` are
            assumed to be ``"input"``.
        skip_dim_check: If ``True``, skip the import-time dimensional
            consistency check.  Useful during migration.

    Returns:
        A decorator that registers the heavy function.

    Raises:
        ValueError: If the expression is dimensionally inconsistent
            (unless ``skip_dim_check=True``).

    Example::

        import sympy as sp

        P, V, n, R, T = sp.symbols("P V n R T")

        @symbolic_atom(
            witness=witness_ideal_gas,
            expr=sp.Eq(P * V, n * R * T),
            dim_map={
                "P": PASCAL,
                "V": VOLUME,
                "n": MOLE,
                "R": JOULE.divide(MOLE).divide(KELVIN),
                "T": KELVIN,
            },
            constants={"R": 8.314},
            bibliography=["Q36253"],
        )
        def ideal_gas_law(P, V, n, T):
            ...
    """
    # Build variable roles
    if variables is None:
        const_names = set(constants or {})
        variables = {
            k: "constant" if k in const_names else "input"
            for k in dim_map
        }

    # Serialize the expression
    srepr_str = serialize_expr(expr)

    # Build the SymbolicExpression
    symbolic = SymbolicExpression(
        srepr_str=srepr_str,
        variables=variables,
        dim_map=dim_map,
        validity_bounds=validity_bounds or {},
        constants=constants or {},
        bibliography=bibliography or [],
    )

    # Import-time dimensional consistency check
    if not skip_dim_check:
        errors = symbolic.check_dimensional_consistency()
        if errors:
            raise ValueError(
                f"Dimensional inconsistency in symbolic atom "
                f"(expression: {srepr_str[:80]}...): "
                + "; ".join(errors)
            )

    # Delegate to register_atom
    def decorator(heavy_func: Callable) -> Callable:
        # Use register_atom to handle the standard registration
        registered = register_atom(
            witness,
            name=name,
            dim_map=dim_map,
        )(heavy_func)

        # Attach symbolic metadata to the registry entry
        from sciona.ghost.registry import REGISTRY
        atom_name = name or heavy_func.__name__
        if atom_name in REGISTRY:
            REGISTRY[atom_name]["symbolic"] = symbolic

        return registered

    return decorator

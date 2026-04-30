"""Cross-atom algebraic simplification pass.

When adjacent atoms in a CDG both have ``SymbolicExpression`` metadata,
the simplifier composes their expressions symbolically and attempts
algebraic reduction via ``sympy.simplify``.  For example:

    Atom A: y = exp(x)
    Atom B: z = ln(y)

After composition and simplification: z = x.  The redundant pair is
collapsed into a single identity (or simpler) expression.

This is an **optional** optimisation pass that runs between assembly and
compilation.  It never changes semantics — only reduces computational cost
and floating-point error accumulation.
"""

from __future__ import annotations

import logging
from typing import Any

from sciona.ghost.registry import REGISTRY
from sciona.ghost.symbolic import SymbolicExpression, _eval_srepr, serialize_expr
from sciona.synthesizer.models import AssemblyUnit, GlueEdge

logger = logging.getLogger(__name__)

# Lazy SymPy import
_sympy = None


def _ensure_sympy():
    global _sympy
    if _sympy is None:
        import sympy as _sp
        _sympy = _sp
    return _sympy


def _get_symbolic(unit: AssemblyUnit) -> SymbolicExpression | None:
    """Look up the SymbolicExpression for an AssemblyUnit from the registry."""
    for key in (unit.declaration_name, unit.name):
        entry = REGISTRY.get(key)
        if entry and entry.get("symbolic"):
            return entry["symbolic"]
    return None


def simplify_pipeline(
    units: list[AssemblyUnit],
    glue_edges: list[GlueEdge],
) -> tuple[list[AssemblyUnit], list[SimplificationResult]]:
    """Attempt algebraic simplification of adjacent symbolic atoms.

    Walks the assembly units in topological order.  For each pair of
    adjacent atoms (A -> B) where both have symbolic expressions and
    the connecting edge maps A's output to B's input:

    1. Compose B(A(x)) symbolically.
    2. Run ``sympy.simplify()`` on the composition.
    3. If the simplified form is strictly simpler (fewer operations),
       replace both units with a single fused unit.

    Args:
        units: Assembly units in topological order.
        glue_edges: Edges connecting them.

    Returns:
        A tuple of (possibly reduced unit list, list of simplification
        results describing what was done).
    """
    sp = _ensure_sympy()

    results: list[SimplificationResult] = []
    unit_map = {u.node_id: u for u in units}

    # Build adjacency: for each unit, which single successor does it feed?
    # Only consider simple linear chains (one output -> one input).
    successor: dict[str, str] = {}
    for edge in glue_edges:
        if edge.source_id in unit_map and edge.target_id in unit_map:
            # Only track if source has exactly one outgoing edge
            if edge.source_id not in successor:
                successor[edge.source_id] = edge.target_id
            else:
                # Multiple successors — don't simplify this source
                successor[edge.source_id] = ""  # sentinel for "skip"

    # Track which units have been fused
    fused: set[str] = set()
    replacements: dict[str, AssemblyUnit] = {}

    for unit in units:
        if unit.node_id in fused:
            continue

        succ_id = successor.get(unit.node_id, "")
        if not succ_id or succ_id in fused:
            continue

        succ_unit = unit_map.get(succ_id)
        if succ_unit is None:
            continue

        sym_a = _get_symbolic(unit)
        sym_b = _get_symbolic(succ_unit)
        if sym_a is None or sym_b is None:
            continue

        # Find the connecting edge to know which variable to substitute
        connecting_edge = None
        for edge in glue_edges:
            if edge.source_id == unit.node_id and edge.target_id == succ_id:
                connecting_edge = edge
                break

        if connecting_edge is None:
            continue

        try:
            result = _try_compose_and_simplify(
                sym_a, sym_b, connecting_edge, unit, succ_unit, sp,
            )
        except Exception as exc:
            logger.debug(
                "Simplification failed for %s -> %s: %s",
                unit.name, succ_unit.name, exc,
            )
            continue

        if result is None:
            continue

        results.append(result)
        fused.add(unit.node_id)
        fused.add(succ_unit.node_id)
        replacements[unit.node_id] = result.fused_unit

    # Build output unit list
    output_units: list[AssemblyUnit] = []
    for unit in units:
        if unit.node_id in fused:
            if unit.node_id in replacements:
                output_units.append(replacements[unit.node_id])
            # else: this was the successor, already covered by the replacement
        else:
            output_units.append(unit)

    return output_units, results


class SimplificationResult:
    """Record of a successful algebraic simplification."""

    def __init__(
        self,
        source_unit: AssemblyUnit,
        target_unit: AssemblyUnit,
        fused_unit: AssemblyUnit,
        original_ops: int,
        simplified_ops: int,
        simplified_srepr: str,
    ):
        self.source_unit = source_unit
        self.target_unit = target_unit
        self.fused_unit = fused_unit
        self.original_ops = original_ops
        self.simplified_ops = simplified_ops
        self.simplified_srepr = simplified_srepr

    def __repr__(self) -> str:
        return (
            f"SimplificationResult({self.source_unit.name} + "
            f"{self.target_unit.name} -> {self.fused_unit.name}, "
            f"ops: {self.original_ops} -> {self.simplified_ops})"
        )


def _count_ops(expr: Any, sp: Any) -> int:
    """Count the number of operations in a SymPy expression."""
    try:
        return int(sp.count_ops(expr))
    except Exception:
        return 999


def _try_compose_and_simplify(
    sym_a: SymbolicExpression,
    sym_b: SymbolicExpression,
    edge: GlueEdge,
    unit_a: AssemblyUnit,
    unit_b: AssemblyUnit,
    sp: Any,
) -> SimplificationResult | None:
    """Try to compose two symbolic expressions and simplify.

    Returns ``None`` if the simplification doesn't reduce complexity.
    """
    expr_a = _eval_srepr(sym_a.srepr_str)
    expr_b = _eval_srepr(sym_b.srepr_str)

    # Handle Eq: use the RHS as the expression body
    if isinstance(expr_a, sp.Eq):
        expr_a = expr_a.rhs
    if isinstance(expr_b, sp.Eq):
        expr_b = expr_b.rhs

    # Determine which variable in B to substitute with A's expression
    # The edge tells us: A's output_name feeds B's input_name
    b_input_sym = sp.Symbol(edge.input_name)

    # Compose: B(A(x)) = B with b_input_sym replaced by expr_a
    composed = expr_b.subs(b_input_sym, expr_a)

    # Dimensioned quantities are necessarily real-valued, so we can safely
    # add real=True assumptions to unlock identities like log(exp(x)) = x
    # that are invalid for arbitrary complex numbers (branch cut).
    # Variables *without* dimensional info (or explicitly dimensionless
    # with no physical meaning) keep their original assumptions — they may
    # legitimately be complex (e.g., wave functions, Fourier coefficients).
    all_dim = {}
    all_dim.update(sym_a.dim_map)
    all_dim.update(sym_b.dim_map)

    real_subs = {}
    for s in composed.free_symbols:
        dim = all_dim.get(s.name)
        # A symbol is real if it has a non-None dimensional annotation
        if dim is not None:
            real_subs[s] = sp.Symbol(s.name, real=True)
    composed_real = composed.subs(real_subs) if real_subs else composed

    # Count ops before and after
    original_ops = _count_ops(expr_a, sp) + _count_ops(expr_b, sp)

    # Simplify
    simplified = sp.simplify(composed_real)
    simplified_ops = _count_ops(simplified, sp)

    # Only keep if strictly simpler
    if simplified_ops >= original_ops:
        return None

    # Build a fused SymbolicExpression
    fused_srepr = serialize_expr(simplified)

    # Merge dim_maps (A's inputs + B's non-substituted inputs)
    merged_dim_map = dict(sym_a.dim_map)
    for k, v in sym_b.dim_map.items():
        if k != edge.input_name:
            merged_dim_map[k] = v

    # Merge variables
    merged_vars = dict(sym_a.variables)
    for k, v in sym_b.variables.items():
        if k != edge.input_name:
            merged_vars[k] = v

    # Merge constants
    merged_constants = dict(sym_a.constants)
    merged_constants.update(sym_b.constants)

    # Merge validity bounds
    merged_bounds = dict(sym_a.validity_bounds)
    for k, v in sym_b.validity_bounds.items():
        if k != edge.input_name:
            merged_bounds[k] = v

    # Merge bibliography
    merged_bib = list(sym_a.bibliography) + [
        b for b in sym_b.bibliography if b not in sym_a.bibliography
    ]

    fused_symbolic = SymbolicExpression(
        srepr_str=fused_srepr,
        variables=merged_vars,
        dim_map=merged_dim_map,
        validity_bounds=merged_bounds,
        constants=merged_constants,
        bibliography=merged_bib,
    )

    # Build a fused AssemblyUnit that takes A's inputs and produces B's outputs
    fused_unit = AssemblyUnit(
        node_id=unit_a.node_id,  # keep A's ID for graph continuity
        name=f"{unit_a.name}__{unit_b.name}_fused",
        declaration_name=f"{unit_a.declaration_name}__fused",
        type_signature=unit_b.type_signature or unit_a.type_signature,
        inputs=unit_a.inputs,
        outputs=unit_b.outputs,
        requires_glue=unit_b.requires_glue,
        tunable_param_names=list(
            set(unit_a.tunable_param_names) | set(unit_b.tunable_param_names)
        ),
    )

    # Store the fused symbolic in the registry transiently
    REGISTRY[fused_unit.declaration_name] = {
        "impl": None,
        "witness": None,
        "doc": f"Fused: {unit_a.name} + {unit_b.name}",
        "signature": {},
        "heavy_signature": {},
        "module": "",
        "name": fused_unit.declaration_name,
        "dim_signature": merged_dim_map,
        "symbolic": fused_symbolic,
    }

    return SimplificationResult(
        source_unit=unit_a,
        target_unit=unit_b,
        fused_unit=fused_unit,
        original_ops=original_ops,
        simplified_ops=simplified_ops,
        simplified_srepr=fused_srepr,
    )

"""Source-neutral normalization for externally discovered symbolic equations.

This module is intentionally adapter-agnostic.  Source workers can hand it
raw candidate-like dicts, Pydantic models, or simple objects; the normalizer
preserves source payload details, parses locally supported SymPy formulas,
computes deterministic hashes, and emits metadata that can instantiate
``SymbolicExpression``.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, Literal

from pydantic import BaseModel, Field

from sciona.ghost.dimensions import DimensionalSignature
from sciona.ghost.symbolic import (
    SymbolicExpression,
    deserialize_expr,
    serialize_expr,
)

ParseStatus = Literal["parsed", "failed"]


class VariableHint(BaseModel):
    """Source-provided metadata for one symbolic variable."""

    symbol: str
    role: str = "input"
    aliases: list[str] = Field(default_factory=list)
    dim_signature: DimensionalSignature | None = None
    quantity_kind: str | None = None
    qudt_uri: str | None = None
    unit_uri: str | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)


class NormalizedVariable(BaseModel):
    """Canonical variable metadata emitted by the normalizer."""

    symbol: str
    role: str
    aliases: list[str] = Field(default_factory=list)
    dim_signature: DimensionalSignature | None = None
    quantity_kind: str | None = None
    qudt_uri: str | None = None
    unit_uri: str | None = None
    assumptions: dict[str, Any] = Field(default_factory=dict)


class ReviewTask(BaseModel):
    """A reviewable normalization issue."""

    code: str
    message: str
    symbol: str | None = None


class NormalizedSymbolicCandidate(BaseModel):
    """Normalized equation candidate plus SymbolicExpression-compatible fields."""

    candidate_id: str | None = None
    source_id: str | None = None
    source_version: str | None = None
    raw_formula: Any = None
    raw_formula_format: str | None = None
    parse_status: ParseStatus
    parse_error: str | None = None
    srepr_str: str | None = None
    expression_hash: str | None = None
    topology_hash: str | None = None
    dimensional_hash: str | None = None
    variables: dict[str, NormalizedVariable] = Field(default_factory=dict)
    constants: dict[str, float] = Field(default_factory=dict)
    validity_bounds: dict[str, tuple[float | None, float | None]] = Field(
        default_factory=dict
    )
    bibliography: list[str] = Field(default_factory=list)
    review_tasks: list[ReviewTask] = Field(default_factory=list)

    def symbolic_expression_kwargs(self) -> dict[str, Any]:
        """Return kwargs accepted by ``SymbolicExpression``.

        Unknown dimensions are omitted from ``dim_map`` so callers can decide
        whether to block publication.  Review tasks identify those gaps.
        """
        if self.srepr_str is None:
            raise ValueError("Cannot build SymbolicExpression kwargs for failed parse")
        return {
            "srepr_str": self.srepr_str,
            "variables": {
                symbol: variable.role for symbol, variable in self.variables.items()
            },
            "dim_map": {
                symbol: variable.dim_signature
                for symbol, variable in self.variables.items()
                if variable.dim_signature is not None
            },
            "validity_bounds": self.validity_bounds,
            "constants": self.constants,
            "bibliography": self.bibliography,
        }

    def to_symbolic_expression(self) -> SymbolicExpression:
        """Instantiate the existing Sciona symbolic model."""
        return SymbolicExpression(**self.symbolic_expression_kwargs())


@dataclass(frozen=True)
class _CandidateFields:
    candidate_id: str | None
    source_id: str | None
    source_version: str | None
    formula: Any
    formula_format: str | None
    variable_hints: Any
    constants: dict[str, float]
    validity_bounds: dict[str, tuple[float | None, float | None]]
    bibliography: list[str]


def normalize_symbolic_candidate(
    candidate: Any,
    *,
    require_dimensions: bool = False,
) -> NormalizedSymbolicCandidate:
    """Normalize a source-agnostic equation candidate.

    Args:
        candidate: Dict/Pydantic/object with candidate-like fields.  Supported
            formula fields include ``expression``, ``formula``, ``raw_formula``,
            ``formula_text``, and ``sympy_expr``.
        require_dimensions: When true, missing variable dimensions are emitted
            as blocking review tasks by using the ``missing_required_dimension``
            code.  The function still returns a normalized candidate so raw
            candidates are retained.
    """
    fields = _extract_candidate_fields(candidate)
    try:
        expr = _parse_formula(fields.formula, fields.formula_format)
    except Exception as exc:  # noqa: BLE001 - retain reviewable parse failure
        return NormalizedSymbolicCandidate(
            candidate_id=fields.candidate_id,
            source_id=fields.source_id,
            source_version=fields.source_version,
            raw_formula=fields.formula,
            raw_formula_format=fields.formula_format,
            parse_status="failed",
            parse_error=str(exc),
            constants=fields.constants,
            validity_bounds=fields.validity_bounds,
            bibliography=fields.bibliography,
            review_tasks=[
                ReviewTask(code="parse_failed", message=str(exc)),
            ],
        )

    srepr_str = serialize_expr(expr)
    expression_hash = _stable_hash("expr", srepr_str)
    topology_hash = _stable_hash("topology", _topology_srepr(expr))

    variable_hints = _normalize_variable_hints(fields.variable_hints)
    variables = _build_variables(expr, variable_hints, fields.constants)
    review_tasks = _review_variable_gaps(
        variables,
        require_dimensions=require_dimensions,
    )
    dim_parts = [
        f"{symbol}:{variable.dim_signature.to_compact()}"
        for symbol, variable in sorted(variables.items())
        if variable.dim_signature is not None
    ]
    dimensional_hash = _stable_hash(
        "dimensional",
        topology_hash + "|" + "|".join(dim_parts),
    )

    return NormalizedSymbolicCandidate(
        candidate_id=fields.candidate_id,
        source_id=fields.source_id,
        source_version=fields.source_version,
        raw_formula=fields.formula,
        raw_formula_format=fields.formula_format,
        parse_status="parsed",
        srepr_str=srepr_str,
        expression_hash=expression_hash,
        topology_hash=topology_hash,
        dimensional_hash=dimensional_hash,
        variables=variables,
        constants=fields.constants,
        validity_bounds=fields.validity_bounds,
        bibliography=fields.bibliography,
        review_tasks=review_tasks,
    )


def _extract_candidate_fields(candidate: Any) -> _CandidateFields:
    data = _as_mapping(candidate)
    formula = _first_present(
        data,
        "expression",
        "sympy_expr",
        "formula",
        "raw_formula",
        "formula_text",
        "srepr_str",
    )
    if formula is None:
        raise ValueError("candidate has no formula/expression field")
    formula_format = _first_present(
        data,
        "formula_format",
        "raw_formula_format",
        "expression_format",
    )
    if formula_format is None and "srepr_str" in data:
        formula_format = "srepr"
    return _CandidateFields(
        candidate_id=_string_or_none(
            _first_present(data, "candidate_id", "equation_id", "id", "external_id")
        ),
        source_id=_string_or_none(_first_present(data, "source_id", "source")),
        source_version=_string_or_none(_first_present(data, "source_version")),
        formula=formula,
        formula_format=_string_or_none(formula_format),
        variable_hints=_first_present(data, "variable_hints", "variables", "symbols"),
        constants=_normalize_constants(_first_present(data, "constants")),
        validity_bounds=_normalize_validity_bounds(
            _first_present(data, "validity_bounds", "bounds")
        ),
        bibliography=_normalize_string_list(
            _first_present(data, "bibliography", "references", "reference_keys")
        ),
    )


def _parse_formula(formula: Any, formula_format: str | None) -> Any:
    sp = _ensure_sympy()
    if isinstance(formula, sp.Basic):
        return formula
    if formula_format and formula_format.lower() in {"srepr", "sympy_srepr"}:
        return deserialize_expr(str(formula))
    if not isinstance(formula, str):
        raise TypeError(f"unsupported formula type: {type(formula).__name__}")

    text = formula.strip()
    if not text:
        raise ValueError("formula is empty")
    if formula_format and formula_format.lower() in {"latex", "mathml"}:
        raise ValueError(
            f"{formula_format} parsing is not supported by this local scaffold"
        )
    if "=" in text and not text.startswith("Eq("):
        lhs, rhs = _split_equation_text(text)
        return sp.Eq(_parse_expr(lhs), _parse_expr(rhs))
    return _parse_expr(text)


def _parse_expr(text: str) -> Any:
    sp = _ensure_sympy()
    from sympy.parsing.sympy_parser import parse_expr, standard_transformations

    return parse_expr(
        text,
        global_dict={"__builtins__": {}, "Symbol": sp.Symbol, "Eq": sp.Eq},
        local_dict=_sympy_local_dict(sp),
        transformations=standard_transformations,
        evaluate=True,
    )


def _split_equation_text(text: str) -> tuple[str, str]:
    if text.count("=") != 1:
        raise ValueError("equation text must contain exactly one '='")
    lhs, rhs = text.split("=", 1)
    if not lhs.strip() or not rhs.strip():
        raise ValueError("equation sides must be non-empty")
    return lhs, rhs


def _topology_srepr(expr: Any) -> str:
    sp = _ensure_sympy()
    symbols = sorted(expr.free_symbols, key=lambda symbol: str(symbol))
    replacements = {
        symbol: sp.Symbol(f"_v{i}", real=symbol.is_real)
        for i, symbol in enumerate(symbols)
    }
    return serialize_expr(expr.xreplace(replacements))


def _build_variables(
    expr: Any,
    hints: dict[str, VariableHint],
    constants: dict[str, float],
) -> dict[str, NormalizedVariable]:
    symbols = sorted(str(symbol) for symbol in expr.free_symbols)
    variables: dict[str, NormalizedVariable] = {}
    for symbol in symbols:
        hint = hints.get(symbol)
        role = "constant" if symbol in constants else "input"
        if hint is not None:
            role = hint.role or role
        variables[symbol] = NormalizedVariable(
            symbol=symbol,
            role=role,
            aliases=_dedupe_preserve_order(
                [symbol, *(hint.aliases if hint else [])]
            ),
            dim_signature=hint.dim_signature if hint else None,
            quantity_kind=hint.quantity_kind if hint else None,
            qudt_uri=hint.qudt_uri if hint else None,
            unit_uri=hint.unit_uri if hint else None,
            assumptions=hint.assumptions if hint else {},
        )
    return variables


def _review_variable_gaps(
    variables: dict[str, NormalizedVariable],
    *,
    require_dimensions: bool,
) -> list[ReviewTask]:
    review_tasks: list[ReviewTask] = []
    for symbol, variable in sorted(variables.items()):
        if variable.dim_signature is None:
            review_tasks.append(
                ReviewTask(
                    code=(
                        "missing_required_dimension"
                        if require_dimensions
                        else "missing_dimension"
                    ),
                    symbol=symbol,
                    message=f"No dimensional signature resolved for symbol {symbol!r}",
                )
            )
    return review_tasks


def _normalize_variable_hints(raw: Any) -> dict[str, VariableHint]:
    if raw is None:
        return {}
    hints: dict[str, VariableHint] = {}
    if isinstance(raw, Mapping):
        iterable = []
        for symbol, value in raw.items():
            if isinstance(value, Mapping):
                item = {"symbol": symbol, **dict(value)}
            elif isinstance(value, DimensionalSignature) or isinstance(value, str):
                item = {"symbol": symbol, "dim_signature": value}
            else:
                item = {"symbol": symbol}
            iterable.append(item)
    elif isinstance(raw, list | tuple):
        iterable = raw
    else:
        raise TypeError("variable hints must be a mapping or list")

    for item in iterable:
        data = dict(_as_mapping(item))
        symbol = str(_first_present(data, "symbol", "name", "id"))
        if not symbol or symbol == "None":
            raise ValueError("variable hint missing symbol/name")
        aliases = _normalize_string_list(_first_present(data, "aliases", "alias"))
        dim_signature = _coerce_dimensional_signature(
            _first_present(data, "dim_signature", "dimension", "dimensions")
        )
        hints[symbol] = VariableHint(
            symbol=symbol,
            role=str(_first_present(data, "role") or "input"),
            aliases=aliases,
            dim_signature=dim_signature,
            quantity_kind=_string_or_none(_first_present(data, "quantity_kind")),
            qudt_uri=_string_or_none(_first_present(data, "qudt_uri")),
            unit_uri=_string_or_none(_first_present(data, "unit_uri")),
            assumptions=dict(_first_present(data, "assumptions") or {}),
        )
    return hints


def _coerce_dimensional_signature(raw: Any) -> DimensionalSignature | None:
    if raw is None:
        return None
    if isinstance(raw, DimensionalSignature):
        return raw
    if isinstance(raw, str):
        return DimensionalSignature.from_compact(raw)
    if isinstance(raw, Mapping):
        return DimensionalSignature(**dict(raw))
    raise TypeError(f"unsupported dimension signature: {type(raw).__name__}")


def _normalize_constants(raw: Any) -> dict[str, float]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("constants must be a mapping")
    return {str(key): float(value) for key, value in raw.items()}


def _normalize_validity_bounds(
    raw: Any,
) -> dict[str, tuple[float | None, float | None]]:
    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise TypeError("validity bounds must be a mapping")
    normalized: dict[str, tuple[float | None, float | None]] = {}
    for symbol, bounds in raw.items():
        if isinstance(bounds, Mapping):
            lower = _first_present(bounds, "min", "lower")
            upper = _first_present(bounds, "max", "upper")
        else:
            lower, upper = bounds
        normalized[str(symbol)] = (_float_or_none(lower), _float_or_none(upper))
    return normalized


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"expected mapping-like value, got {type(value).__name__}")


def _first_present(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _normalize_string_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return [raw]
    return _dedupe_preserve_order(str(item) for item in raw)


def _dedupe_preserve_order(values: Any) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            result.append(value)
            seen.add(value)
    return result


def _string_or_none(value: Any) -> str | None:
    return None if value is None else str(value)


def _float_or_none(value: Any) -> float | None:
    return None if value is None else float(value)


def _stable_hash(namespace: str, payload: str) -> str:
    return sha256(f"{namespace}\0{payload}".encode("utf-8")).hexdigest()


def _ensure_sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "SymPy >= 1.12 is required for symbolic equation normalization"
        ) from exc
    return sp


def _sympy_local_dict(sp: Any) -> dict[str, Any]:
    local_dict = {
        "Eq": sp.Eq,
        "Derivative": sp.Derivative,
        "Integral": sp.Integral,
        "sqrt": sp.sqrt,
        "exp": sp.exp,
        "log": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "pi": sp.pi,
        "E": sp.E,
    }
    return local_dict

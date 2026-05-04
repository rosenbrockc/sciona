"""Side-effect-free symbolic normalization for physics ingest candidates."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict

from sciona.ghost.symbolic import deserialize_expr, serialize_expr
from sciona.ghost.symbolic_normalization import (
    NormalizedSymbolicCandidate,
    normalize_symbolic_candidate,
)
from sciona.physics_ingest.staging import SymbolicExpressionRow


_SUPPORTED_SOURCE_FORMATS = {
    "",
    "latex",
    "sympy",
    "plain_text",
}
_DIMENSION_REVIEW_TASK_CODES = {
    "missing_dimension",
    "missing_required_dimension",
}


class NormalizationDiagnostic(BaseModel):
    """Reviewable diagnostic emitted while normalizing a candidate formula."""

    model_config = ConfigDict(frozen=True)

    code: str
    message: str
    severity: str = "info"
    symbol: str | None = None


@dataclass(frozen=True)
class NormalizedExpressionDraft:
    """A staged expression row draft plus normalization diagnostics."""

    row: SymbolicExpressionRow
    normalized_candidate: NormalizedSymbolicCandidate
    diagnostics: tuple[NormalizationDiagnostic, ...] = ()

    def to_insert_dict(self) -> dict[str, Any]:
        """Return the JSON-ready row dictionary for insertion planning."""

        return self.row.to_insert_dict()


@dataclass(frozen=True)
class QudtAssistedCandidate:
    """Candidate copy enriched with reviewable QUDT dimension diagnostics."""

    candidate: dict[str, Any]
    diagnostics: tuple[NormalizationDiagnostic, ...] = ()


def normalize_candidate_expression_draft(
    candidate: Any,
    *,
    artifact_id: str,
    version_id: str,
    expression_kind: str = "equation",
    expression_role: str = "primary",
    require_dimensions: bool = False,
) -> NormalizedExpressionDraft:
    """Normalize one raw candidate into a validated symbolic expression draft.

    The helper is intentionally side-effect free: it does not query a database,
    mutate the input candidate, or drop failed parses.  Failed candidates return
    a ``parse_failed`` expression row with ``needs_human`` review status and
    diagnostics in ``evidence_json``.
    """

    return _normalize_candidate_expression_draft(
        candidate,
        artifact_id=artifact_id,
        version_id=version_id,
        expression_kind=expression_kind,
        expression_role=expression_role,
        require_dimensions=require_dimensions,
        initial_diagnostics=(),
    )


def normalize_candidate_expression_draft_with_qudt_dimensions(
    candidate: Any,
    *,
    qudt_records: Iterable[Any],
    artifact_id: str,
    version_id: str,
    expression_kind: str = "equation",
    expression_role: str = "primary",
    require_dimensions: bool = False,
) -> NormalizedExpressionDraft:
    """Normalize one candidate after side-effect-free QUDT dimension resolution.

    Only uniquely resolved QUDT records fill missing variable dimensions.  Empty
    or unknown placeholders that QUDT cannot resolve are omitted from the copied
    candidate before normalization so they remain reviewable as missing
    dimensions instead of being interpreted as dimensionless.
    """

    assisted = resolve_candidate_variable_dimensions_from_qudt(
        candidate,
        qudt_records=qudt_records,
    )
    return _normalize_candidate_expression_draft(
        assisted.candidate,
        artifact_id=artifact_id,
        version_id=version_id,
        expression_kind=expression_kind,
        expression_role=expression_role,
        require_dimensions=require_dimensions,
        initial_diagnostics=assisted.diagnostics,
    )


def normalize_candidate_expression_drafts_with_qudt_dimensions(
    candidates: Iterable[Any],
    *,
    qudt_records: Iterable[Any],
    artifact_id: str,
    version_id: str,
    expression_kind: str = "equation",
    expression_role: str = "primary",
    require_dimensions: bool = False,
) -> tuple[NormalizedExpressionDraft, ...]:
    """Normalize candidate rows after QUDT-assisted dimension resolution."""

    records = tuple(qudt_records)
    return tuple(
        normalize_candidate_expression_draft_with_qudt_dimensions(
            candidate,
            qudt_records=records,
            artifact_id=artifact_id,
            version_id=version_id,
            expression_kind=expression_kind,
            expression_role=expression_role,
            require_dimensions=require_dimensions,
        )
        for candidate in candidates
    )


def resolve_candidate_variable_dimensions_from_qudt(
    candidate: Any,
    *,
    qudt_records: Iterable[Any],
) -> QudtAssistedCandidate:
    """Return a candidate copy with uniquely resolved QUDT dimensions applied."""

    source = _as_mapping(candidate)
    data = dict(source)
    variable_key = _first_present_key(data, "variable_hints", "variables", "symbols")
    if variable_key is None:
        return QudtAssistedCandidate(candidate=data)

    records = _coerce_qudt_records(qudt_records)
    variables, restore = _copy_variable_hints(data[variable_key])
    diagnostics: list[NormalizationDiagnostic] = []
    for variable in variables:
        symbol = _text(_first_present(variable, "symbol", "name", "id"))
        if not symbol:
            continue
        if not _is_missing_dimension_hint(
            _first_present(variable, "dim_signature", "dimension", "dimensions")
        ):
            continue

        resolution = _resolve_qudt_record_for_variable(variable, records)
        if resolution["status"] == "resolved":
            record = resolution["record"]
            dimension = record.dimension
            variable["dim_signature"] = dimension.compact
            if record.resource_kind == "unit":
                variable.setdefault("unit_uri", record.source_entity_uri)
                variable.setdefault("unit_label", record.source_label)
            if record.resource_kind == "quantity_kind":
                variable.setdefault("qudt_uri", record.source_entity_uri)
                variable.setdefault("quantity_kind", record.source_label)
            diagnostics.append(
                NormalizationDiagnostic(
                    code="qudt_dimension_resolved",
                    message=(
                        f"Resolved QUDT dimension for symbol {symbol!r}: "
                        f"{dimension.compact}"
                    ),
                    symbol=symbol,
                )
            )
            continue

        _drop_dimension_hint(variable)
        diagnostics.append(
            NormalizationDiagnostic(
                code=f"qudt_dimension_{resolution['status']}",
                message=str(resolution["message"]),
                severity="warning",
                symbol=symbol,
            )
        )

    data[variable_key] = restore(variables)
    return QudtAssistedCandidate(candidate=data, diagnostics=tuple(diagnostics))


def _normalize_candidate_expression_draft(
    candidate: Any,
    *,
    artifact_id: str,
    version_id: str,
    expression_kind: str,
    expression_role: str,
    require_dimensions: bool,
    initial_diagnostics: Iterable[NormalizationDiagnostic],
) -> NormalizedExpressionDraft:
    source = _as_mapping(candidate)
    raw_formula = _text(_first_present(source, "raw_formula", "formula", "expression"))
    raw_format = _normalize_formula_format(
        _text(
            _first_present(
                source,
                "raw_formula_format",
                "formula_format",
                "expression_format",
            )
        )
    )
    diagnostics: list[NormalizationDiagnostic] = list(initial_diagnostics)
    normalized_input = _candidate_for_symbolic_normalizer(
        source,
        raw_formula=raw_formula,
        raw_formula_format=raw_format,
        diagnostics=diagnostics,
    )
    normalized = normalize_symbolic_candidate(
        normalized_input,
        require_dimensions=require_dimensions,
    )
    diagnostics.extend(_review_task_diagnostics(normalized))

    roundtrip = _parse_roundtrip_evidence(normalized)
    if roundtrip["status"] != "passed":
        diagnostics.append(
            NormalizationDiagnostic(
                code="parse_roundtrip_failed",
                message=str(roundtrip.get("message") or "parse roundtrip did not pass"),
                severity="error",
            )
        )

    parsed = normalized.parse_status == "parsed" and roundtrip["status"] == "passed"
    review_status = (
        "automated_pass"
        if parsed and not _has_reviewable_diagnostics(diagnostics)
        else "needs_human"
    )
    parse_status = "normalized" if parsed else "parse_failed"
    parse_confidence = _parse_confidence(source, parsed=parsed, diagnostics=diagnostics)
    evidence_json = _evidence_json(
        normalized=normalized,
        diagnostics=diagnostics,
        roundtrip=roundtrip,
        source_format=raw_format,
    )

    row = SymbolicExpressionRow.model_validate(
        {
            "artifact_id": artifact_id,
            "version_id": version_id,
            "candidate_id": _uuid_or_none(_first_present(source, "candidate_id", "id")),
            "expression_kind": expression_kind,
            "expression_role": expression_role,
            "sympy_srepr": normalized.srepr_str or "",
            "canonical_expr_hash": normalized.expression_hash or "",
            "topology_hash": normalized.topology_hash or "",
            "dimensional_hash": normalized.dimensional_hash or "",
            "raw_formula": raw_formula,
            "raw_formula_format": raw_format,
            "source_expression_id": _source_expression_id(source),
            "parse_status": parse_status,
            "parse_confidence": parse_confidence,
            "review_status": review_status,
            "validation_status": "unknown",
            "mechanism_tags": _string_list(source.get("mechanism_tags")),
            "behavioral_archetypes": _string_list(source.get("behavioral_archetypes")),
            "evidence_json": evidence_json,
        }
    )
    return NormalizedExpressionDraft(
        row=row,
        normalized_candidate=normalized,
        diagnostics=tuple(diagnostics),
    )


def normalize_candidate_expression_drafts(
    candidates: Iterable[Any],
    *,
    artifact_id: str,
    version_id: str,
    expression_kind: str = "equation",
    expression_role: str = "primary",
    require_dimensions: bool = False,
) -> tuple[NormalizedExpressionDraft, ...]:
    """Normalize candidate rows without dropping parse failures."""

    return tuple(
        normalize_candidate_expression_draft(
            candidate,
            artifact_id=artifact_id,
            version_id=version_id,
            expression_kind=expression_kind,
            expression_role=expression_role,
            require_dimensions=require_dimensions,
        )
        for candidate in candidates
    )


def _coerce_qudt_records(records: Iterable[Any]) -> tuple[Any, ...]:
    from sciona.physics_ingest.sources.qudt import (
        QudtResourceRecord,
        extract_qudt_resource_record,
    )

    coerced = []
    for record in records:
        if isinstance(record, QudtResourceRecord):
            coerced.append(record)
        elif isinstance(record, Mapping):
            coerced.append(extract_qudt_resource_record(dict(record)))
    return tuple(coerced)


def _copy_variable_hints(value: Any) -> tuple[list[dict[str, Any]], Any]:
    if isinstance(value, Mapping):
        original_keys: list[str] = []
        variables: list[dict[str, Any]] = []
        for symbol, raw_variable in value.items():
            original_keys.append(str(symbol))
            if isinstance(raw_variable, Mapping):
                variable = dict(raw_variable)
            elif isinstance(raw_variable, str):
                variable = {"dim_signature": raw_variable}
            else:
                variable = {}
            variable.setdefault("symbol", str(symbol))
            variables.append(variable)

        def restore_mapping(items: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
            return {
                symbol: dict(item)
                for symbol, item in zip(original_keys, items, strict=True)
            }

        return variables, restore_mapping

    if isinstance(value, list | tuple):
        variables = []
        for raw_variable in value:
            if isinstance(raw_variable, Mapping):
                variables.append(dict(raw_variable))
            else:
                variables.append({"symbol": str(raw_variable)})

        def restore_list(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
            return [dict(item) for item in items]

        return variables, restore_list

    return [], lambda items: list(items)


def _is_missing_dimension_hint(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip().casefold() in {"", "?", "unknown", "unresolved"}
    is_unknown = getattr(value, "is_unknown", False)
    return bool(is_unknown)


def _drop_dimension_hint(variable: dict[str, Any]) -> None:
    for key in ("dim_signature", "dimension", "dimensions"):
        variable.pop(key, None)


def _resolve_qudt_record_for_variable(
    variable: Mapping[str, Any],
    records: tuple[Any, ...],
) -> dict[str, Any]:
    for matches in _qudt_match_tiers(variable, records):
        if not matches:
            continue
        resolved = [record for record in matches if record.dimension is not None]
        compact_dimensions = sorted({record.dimension.compact for record in resolved})
        symbol = _text(_first_present(variable, "symbol", "name", "id"))
        if len(compact_dimensions) == 1:
            return {"status": "resolved", "record": resolved[0]}
        if len(compact_dimensions) > 1:
            return {
                "status": "ambiguous",
                "message": (
                    f"QUDT matches for symbol {symbol!r} have conflicting "
                    f"dimensions: {', '.join(compact_dimensions)}"
                ),
            }
        return {
            "status": "unresolved",
            "message": f"QUDT matches for symbol {symbol!r} have no resolved dimension",
        }
    symbol = _text(_first_present(variable, "symbol", "name", "id"))
    return {
        "status": "unresolved",
        "message": f"No QUDT dimension match for symbol {symbol!r}",
    }


def _qudt_match_tiers(
    variable: Mapping[str, Any],
    records: tuple[Any, ...],
) -> tuple[list[Any], list[Any]]:
    unit_uri = _casefolded(_first_present(variable, "unit_uri", "unit_qudt_uri"))
    quantity_kind_uri = _casefolded(
        _first_present(variable, "quantity_kind_uri", "qudt_uri")
    )
    uri_matches = []
    for record in records:
        record_uri = _casefolded(record.source_entity_uri)
        record_unit_uris = {_casefolded(uri) for uri in record.unit_uris}
        record_quantity_kind_uris = {
            _casefolded(uri) for uri in record.quantity_kind_uris
        }
        if unit_uri and (unit_uri == record_uri or unit_uri in record_unit_uris):
            uri_matches.append(record)
        elif quantity_kind_uri and (
            quantity_kind_uri == record_uri
            or quantity_kind_uri in record_quantity_kind_uris
        ):
            uri_matches.append(record)

    symbol_candidates = {
        _casefolded(_first_present(variable, key))
        for key in (
            "source_symbol",
            "symbol",
            "symbol_name",
            "name",
            "unit",
            "unit_label",
            "quantity_kind",
            "quantity_kind_label",
        )
    }
    symbol_candidates.discard("")
    label_matches = []
    for record in records:
        labels = {
            _casefolded(record.symbol),
            _casefolded(record.source_label),
            _casefolded(record.source_entity_uri.rsplit("/", 1)[-1]),
        }
        labels.discard("")
        if symbol_candidates & labels:
            label_matches.append(record)
    return uri_matches, label_matches


def _candidate_for_symbolic_normalizer(
    source: Mapping[str, Any],
    *,
    raw_formula: str,
    raw_formula_format: str,
    diagnostics: list[NormalizationDiagnostic],
) -> dict[str, Any]:
    data = dict(source)
    data["raw_formula"] = raw_formula
    data["raw_formula_format"] = raw_formula_format
    if raw_formula_format in {"latex", "plain_text"}:
        try:
            parsed = _parse_formula_text(raw_formula, raw_formula_format)
        except Exception as exc:  # noqa: BLE001 - keep failure reviewable below
            parsed = None
            diagnostics.append(
                NormalizationDiagnostic(
                    code=f"{raw_formula_format}_preparse_failed",
                    message=str(exc),
                    severity="warning",
                )
            )
        if parsed is not None:
            data["sympy_expr"] = parsed
            data["formula_format"] = "sympy"
            diagnostics.append(
                NormalizationDiagnostic(
                    code=f"{raw_formula_format}_parsed_locally",
                    message=f"Parsed {raw_formula_format} formula with local normalizer",
                )
            )
    return data


def _parse_formula_text(raw_formula: str, raw_formula_format: str) -> Any | None:
    if not raw_formula:
        return None
    sp = _ensure_sympy()
    if raw_formula_format == "latex":
        parsed = _parse_latex(raw_formula)
        if parsed is not None:
            return parsed
        text = _latex_to_plain_text(raw_formula)
    else:
        text = raw_formula
    return _parse_plain_text(text, sp)


def _parse_latex(raw_formula: str) -> Any | None:
    try:
        from sympy.parsing.latex import parse_latex

        return parse_latex(raw_formula)
    except Exception:  # noqa: BLE001 - optional parser dependencies vary locally
        return None


def _parse_plain_text(text: str, sp: Any) -> Any:
    from sympy.parsing.sympy_parser import (
        convert_xor,
        implicit_multiplication_application,
        parse_expr,
        standard_transformations,
    )

    transformations = standard_transformations + (
        convert_xor,
        implicit_multiplication_application,
    )
    if "=" in text and not text.strip().startswith("Eq("):
        lhs, rhs = _split_equation_text(text)
        return sp.Eq(
            parse_expr(lhs, transformations=transformations, evaluate=True),
            parse_expr(rhs, transformations=transformations, evaluate=True),
        )
    return parse_expr(text, transformations=transformations, evaluate=True)


def _latex_to_plain_text(raw_formula: str) -> str:
    text = raw_formula.strip()
    while r"\frac" in text:
        replaced = _replace_first_frac(text)
        if replaced == text:
            break
        text = replaced
    replacements = {
        r"\cdot": "*",
        r"\times": "*",
        r"\left": "",
        r"\right": "",
        "{": "(",
        "}": ")",
        "^": "**",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text.replace("\\", "")


def _replace_first_frac(text: str) -> str:
    start = text.find(r"\frac")
    if start < 0:
        return text
    numerator_start = text.find("{", start)
    if numerator_start < 0:
        return text
    numerator, numerator_end = _balanced_group(text, numerator_start)
    denominator_start = text.find("{", numerator_end + 1)
    if denominator_start < 0:
        return text
    denominator, denominator_end = _balanced_group(text, denominator_start)
    return (
        text[:start] + f"(({numerator})/({denominator}))" + text[denominator_end + 1 :]
    )


def _balanced_group(text: str, start: int) -> tuple[str, int]:
    depth = 0
    for index in range(start, len(text)):
        char = text[index]
        if char == "{":
            depth += 1
            if depth == 1:
                content_start = index + 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[content_start:index], index
    raise ValueError("unbalanced LaTeX group")


def _split_equation_text(text: str) -> tuple[str, str]:
    if text.count("=") != 1:
        raise ValueError("formula must contain exactly one '='")
    lhs, rhs = text.split("=", 1)
    if not lhs.strip() or not rhs.strip():
        raise ValueError("formula sides must be non-empty")
    return lhs, rhs


def _parse_roundtrip_evidence(
    normalized: NormalizedSymbolicCandidate,
) -> dict[str, Any]:
    if normalized.srepr_str is None:
        return {
            "status": "failed",
            "message": normalized.parse_error or "formula did not parse",
        }
    try:
        reparsed = deserialize_expr(normalized.srepr_str)
        roundtripped = serialize_expr(reparsed)
    except Exception as exc:  # noqa: BLE001 - diagnostic only
        return {"status": "failed", "message": str(exc)}
    return {
        "status": "passed" if roundtripped == normalized.srepr_str else "failed",
        "input_srepr_sha256": normalized.expression_hash,
        "roundtrip_srepr_matches": roundtripped == normalized.srepr_str,
    }


def _evidence_json(
    *,
    normalized: NormalizedSymbolicCandidate,
    diagnostics: list[NormalizationDiagnostic],
    roundtrip: Mapping[str, Any],
    source_format: str,
) -> dict[str, Any]:
    review_task_codes = sorted(task.code for task in normalized.review_tasks)
    review_task_code_counts = {
        code: review_task_codes.count(code) for code in sorted(set(review_task_codes))
    }
    return {
        "normalization": {
            "source_format": source_format,
            "canonical_expr_hash": normalized.expression_hash,
            "topology_hash": normalized.topology_hash,
            "parse_error": normalized.parse_error,
            "review_tasks": [
                task.model_dump(mode="json") for task in normalized.review_tasks
            ],
            "review_task_codes": review_task_codes,
            "review_task_code_counts": review_task_code_counts,
            "dimensions": _dimension_evidence(normalized),
        },
        "parse_roundtrip": dict(roundtrip),
        "diagnostics": [
            diagnostic.model_dump(mode="json") for diagnostic in diagnostics
        ],
    }


def _dimension_evidence(normalized: NormalizedSymbolicCandidate) -> dict[str, Any]:
    provided_signatures = []
    rational_signatures = []
    unknown_symbols = {
        task.symbol
        for task in normalized.review_tasks
        if task.code in _DIMENSION_REVIEW_TASK_CODES and task.symbol
    }
    unknown_review_task_codes = sorted(
        task.code
        for task in normalized.review_tasks
        if task.code in _DIMENSION_REVIEW_TASK_CODES
    )
    for symbol, variable in sorted(normalized.variables.items()):
        dim_signature = variable.dim_signature
        if dim_signature is None:
            unknown_symbols.add(symbol)
            continue
        compact = dim_signature.to_compact()
        if dim_signature.is_unknown:
            unknown_symbols.add(symbol)
        entry = {
            "symbol": symbol,
            "dim_signature": compact,
            "is_unknown": dim_signature.is_unknown,
            "is_rational": _has_rational_exponent(dim_signature),
        }
        provided_signatures.append(entry)
        if entry["is_rational"]:
            rational_signatures.append(entry)

    unknown_symbols_list = sorted(unknown_symbols)
    unknown_review_task_code_counts = {
        code: unknown_review_task_codes.count(code)
        for code in sorted(set(unknown_review_task_codes))
    }
    return {
        "unknown_dimensions": {
            "symbols": unknown_symbols_list,
            "count": len(unknown_symbols_list),
            "review_task_codes": unknown_review_task_codes,
            "review_task_code_counts": unknown_review_task_code_counts,
        },
        "provided_dimensions": {
            "symbols": [entry["symbol"] for entry in provided_signatures],
            "count": len(provided_signatures),
            "signatures": provided_signatures,
        },
        "rational_dimensions": {
            "symbols": [entry["symbol"] for entry in rational_signatures],
            "count": len(rational_signatures),
            "signatures": rational_signatures,
        },
    }


def _has_rational_exponent(dim_signature: Any) -> bool:
    for field in ("M", "L", "T", "I", "Theta", "N", "J"):
        exponent = getattr(dim_signature, field, 0)
        denominator = getattr(exponent, "denominator", 1)
        if denominator != 1:
            return True
    return False


def _review_task_diagnostics(
    normalized: NormalizedSymbolicCandidate,
) -> list[NormalizationDiagnostic]:
    return [
        NormalizationDiagnostic(
            code=task.code,
            message=task.message,
            severity="warning" if task.code.startswith("missing_") else "error",
            symbol=task.symbol,
        )
        for task in normalized.review_tasks
    ]


def _parse_confidence(
    source: Mapping[str, Any],
    *,
    parsed: bool,
    diagnostics: list[NormalizationDiagnostic],
) -> float:
    raw_confidence = source.get("parse_confidence")
    if raw_confidence is not None:
        return max(0.0, min(1.0, float(raw_confidence)))
    if not parsed:
        return 0.0
    if any(diagnostic.severity == "error" for diagnostic in diagnostics):
        return 0.5
    if any(diagnostic.severity == "warning" for diagnostic in diagnostics):
        return 0.75
    return 0.95


def _has_reviewable_diagnostics(
    diagnostics: Iterable[NormalizationDiagnostic],
) -> bool:
    return any(
        diagnostic.severity in {"warning", "error"} for diagnostic in diagnostics
    )


def _normalize_formula_format(value: str) -> str:
    normalized = value.strip().lower()
    if normalized in {"", "plain", "text"}:
        normalized = "plain_text"
    if normalized in {"sympy_srepr", "srepr"}:
        normalized = "sympy"
    if normalized not in _SUPPORTED_SOURCE_FORMATS:
        return "plain_text"
    return normalized


def _source_expression_id(source: Mapping[str, Any]) -> str:
    return _text(
        _first_present(
            source,
            "source_expression_id",
            "source_candidate_id",
            "candidate_id",
            "equation_id",
            "id",
        )
    )


def _uuid_or_none(value: Any) -> str | None:
    if value in (None, ""):
        return None
    try:
        return str(UUID(str(value)))
    except ValueError:
        return None


def _string_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _as_mapping(value: Any) -> Mapping[str, Any]:
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, Mapping):
        return value
    if hasattr(value, "__dict__"):
        return vars(value)
    raise TypeError(f"expected mapping-like candidate, got {type(value).__name__}")


def _first_present(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return None


def _first_present_key(data: Mapping[str, Any], *keys: str) -> str | None:
    for key in keys:
        if key in data and data[key] is not None:
            return key
    return None


def _text(value: Any) -> str:
    return "" if value is None else str(value)


def _casefolded(value: Any) -> str:
    return str(value or "").strip().casefold()


def _ensure_sympy() -> Any:
    try:
        import sympy as sp
    except ImportError as exc:
        raise ImportError(
            "SymPy >= 1.12 is required for symbolic equation normalization"
        ) from exc
    return sp

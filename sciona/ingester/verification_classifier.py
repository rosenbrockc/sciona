"""Deterministic classification for ingest verification failures."""

from __future__ import annotations

import re
from typing import Any

from sciona.ingester.deterministic_ghost_fixer import build_deterministic_ghost_fixes
from sciona.ingester.deterministic_type_fixer import build_deterministic_type_fixes

_TYPE_ERROR_LINE_RE = re.compile(r"error:", re.IGNORECASE)
_TYPE_IMPORT_RE = re.compile(
    r"(No module named|Cannot find implementation or library stub for module named|Name \"\w+\" is not defined)",
    re.IGNORECASE,
)
_TYPE_RETURN_RE = re.compile(r"Incompatible return value type", re.IGNORECASE)
_TYPE_SIGNATURE_RE = re.compile(
    r"(Too many arguments|Too few arguments|Missing positional argument|Unexpected keyword argument|Argument \d+ .* incompatible type)",
    re.IGNORECASE,
)
_TYPE_OUTPUT_RE = re.compile(
    r"(has no attribute|Item \".+\" of .+ has no attribute|Unsupported operand types?|Value of type .+ is not indexable)",
    re.IGNORECASE,
)
_TYPE_STATE_RE = re.compile(
    r"(possibly unbound|may be unbound|None\" has no attribute|Optional.+has no attribute|union-attr)",
    re.IGNORECASE,
)
_TYPE_QUERY_RE = re.compile(r"(mutation|mutates?)", re.IGNORECASE)

_GHOST_QUERY_RE = re.compile(r"(query|metadata)", re.IGNORECASE)
_GHOST_OUTPUT_RE = re.compile(
    r"(domain mismatch|shape mismatch|attribute.*missing|has no attribute)",
    re.IGNORECASE,
)
_GHOST_STATE_RE = re.compile(
    r"(state|rehydrat|fitted|config)",
    re.IGNORECASE,
)


def _failure(
    *,
    stage: str,
    reason_code: str,
    repairable: bool,
    route: str,
    summary: str,
    issues: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "reason_code": reason_code,
        "repairable": repairable,
        "route": route,
        "summary": summary,
        "issues": issues,
    }


def _type_issue(
    *,
    reason_code: str,
    repairable: bool,
    error_line: str,
) -> dict[str, Any]:
    return {
        "reason_code": reason_code,
        "repairable": repairable,
        "error_line": error_line,
    }


def classify_type_failure(
    mypy_errors: str,
    *,
    source_files: dict[str, str],
) -> dict[str, Any]:
    """Classify mypy failures into repairable mechanical vs semantic failures."""
    error_lines = [
        line.strip() for line in mypy_errors.splitlines() if _TYPE_ERROR_LINE_RE.search(line)
    ]
    if not error_lines:
        return _failure(
            stage="verify_types",
            reason_code="unknown_or_unclassified",
            repairable=False,
            route="end",
            summary="No mypy error lines could be classified",
            issues=[],
        )

    deterministic_patches = build_deterministic_type_fixes(mypy_errors, source_files)
    issues: list[dict[str, Any]] = []
    repairable = deterministic_patches is not None
    mechanical_reason: str | None = None

    for error_line in error_lines:
        if _TYPE_SIGNATURE_RE.search(error_line):
            issues.append(
                _type_issue(
                    reason_code="semantic_signature",
                    repairable=False,
                    error_line=error_line,
                )
            )
            repairable = False
            continue
        if _TYPE_OUTPUT_RE.search(error_line):
            issues.append(
                _type_issue(
                    reason_code="semantic_output_binding",
                    repairable=False,
                    error_line=error_line,
                )
            )
            repairable = False
            continue
        if _TYPE_STATE_RE.search(error_line):
            issues.append(
                _type_issue(
                    reason_code="semantic_state_rehydration",
                    repairable=False,
                    error_line=error_line,
                )
            )
            repairable = False
            continue
        if _TYPE_QUERY_RE.search(error_line):
            issues.append(
                _type_issue(
                    reason_code="semantic_query_mutation",
                    repairable=False,
                    error_line=error_line,
                )
            )
            repairable = False
            continue
        if _TYPE_IMPORT_RE.search(error_line):
            issues.append(
                _type_issue(
                    reason_code="mechanical_import",
                    repairable=repairable,
                    error_line=error_line,
                )
            )
            mechanical_reason = mechanical_reason or "mechanical_import"
            continue
        if _TYPE_RETURN_RE.search(error_line):
            issues.append(
                _type_issue(
                    reason_code="mechanical_annotation",
                    repairable=repairable,
                    error_line=error_line,
                )
            )
            mechanical_reason = mechanical_reason or "mechanical_annotation"
            continue
        issues.append(
            _type_issue(
                reason_code="unknown_or_unclassified",
                repairable=False,
                error_line=error_line,
            )
        )
        repairable = False

    non_repairable = next((issue for issue in issues if not issue["repairable"]), None)
    if non_repairable is not None:
        return _failure(
            stage="verify_types",
            reason_code=str(non_repairable["reason_code"]),
            repairable=False,
            route="end",
            summary="Type verification found a non-repairable semantic or unknown failure",
            issues=issues,
        )

    return _failure(
        stage="verify_types",
        reason_code=mechanical_reason or "unknown_or_unclassified",
        repairable=True,
        route="repair_types",
        summary="Type verification found a deterministic mechanical failure",
        issues=issues,
    )


def classify_ghost_failure(
    report: dict[str, Any],
    *,
    witness_source: str,
) -> dict[str, Any]:
    """Classify ghost simulation failures into repairable vs fail-fast buckets."""
    if report.get("cyclic_deadlock"):
        return _failure(
            stage="verify_ghost",
            reason_code="message_cycle",
            repairable=True,
            route="repair_message_cycle",
            summary="Ghost simulation found a repairable message-passing cycle",
            issues=[
                {
                    "reason_code": "message_cycle",
                    "repairable": True,
                    "error_line": str(report.get("error", "") or "cyclic deadlock"),
                }
            ],
        )

    error_message = str(report.get("error", "") or "")
    error_node = str(report.get("error_node", "") or "")
    error_function = str(report.get("error_function", "") or "")
    deterministic_fixes = build_deterministic_ghost_fixes(
        error_node,
        error_function,
        error_message,
        witness_source,
    )
    if deterministic_fixes is not None:
        return _failure(
            stage="verify_ghost",
            reason_code="mechanical_reference",
            repairable=True,
            route="repair_ghost",
            summary="Ghost simulation found a deterministic witness stub failure",
            issues=[
                {
                    "reason_code": "mechanical_reference",
                    "repairable": True,
                    "error_line": error_message,
                }
            ],
        )

    if _GHOST_QUERY_RE.search(error_function) or _GHOST_QUERY_RE.search(error_node):
        reason_code = "semantic_query_mutation"
    elif _GHOST_STATE_RE.search(error_message):
        reason_code = "semantic_state_rehydration"
    elif _GHOST_OUTPUT_RE.search(error_message):
        reason_code = "semantic_output_binding"
    else:
        reason_code = "unknown_or_unclassified"

    return _failure(
        stage="verify_ghost",
        reason_code=reason_code,
        repairable=False,
        route="end",
        summary="Ghost simulation found a non-repairable semantic or unknown failure",
        issues=[
            {
                "reason_code": reason_code,
                "repairable": False,
                "error_line": error_message,
            }
        ],
    )

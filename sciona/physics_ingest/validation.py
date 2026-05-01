"""Offline validation for physics symbolic fixtures and PDG CDG rows."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from sciona.physics_ingest.pdg_cdg import (
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
    validate_pdg_cdg_publication_graph,
)
from sciona.physics_ingest.publication import load_symbolic_publication_manifest
from sciona.physics_ingest.sources import (
    build_physics_source_retrieval_run_plan,
    build_source_execution_readiness_report,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document


VALIDATION_REPORT_KIND = "physics_ingestion_validation"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PDG_FIXTURE_GLOBS = (
    Path("tests") / "physics_ingest" / "fixtures" / "pdg_payloads" / "*.pdg.json",
    Path("docs") / "physics_ingest" / "fixtures" / "pdg_payloads" / "*.pdg.json",
)
_ARTIFACT_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.validation.artifact")
_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.validation.version")
_EXPRESSION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.validation.expression",
)


@dataclass(frozen=True)
class ValidationIssue:
    """A stable validation issue suitable for CI reports."""

    reason: str
    severity: str = "error"
    detail: str = ""
    table: str = ""
    subject: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason": self.reason,
            "severity": self.severity,
            "detail": self.detail,
            "table": self.table,
            "subject": self.subject,
        }


@dataclass(frozen=True)
class ValidationCheck:
    """One validation check and its issues."""

    check_id: str
    subject: str
    issues: tuple[ValidationIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not any(issue.severity == "error" for issue in self.issues)

    def to_dict(self) -> dict[str, Any]:
        return {
            "check_id": self.check_id,
            "subject": self.subject,
            "ok": self.ok,
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": dict(self.metadata),
        }


def build_physics_ingestion_validation_report(
    *,
    fixture_paths: Iterable[Path | str] = (),
    atoms_repo: Path | str | None = None,
    pdg_payload_paths: Iterable[Path | str] = (),
    include_default_pdg: bool = True,
    include_source_execution: bool = True,
    source_retrieval_run_plan: Any | None = None,
    source_max_jobs: int | None = None,
    source_job_id: str | Iterable[str] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Build a JSON-safe offline validation report.

    The report performs no database or network IO. When ``atoms_repo`` is
    available, fixture JSON is compared with the live symbolic publication
    manifest for the modules named by the fixture.
    """

    checks: list[ValidationCheck] = []
    fixture_path_list = tuple(Path(path) for path in fixture_paths)
    pdg_path_list = tuple(Path(path) for path in pdg_payload_paths)
    live_manifest_builder = _load_live_manifest_builder(atoms_repo)

    if strict and not fixture_path_list:
        checks.append(
            ValidationCheck(
                check_id="symbolic_fixture_inventory",
                subject="symbolic_publication_fixtures",
                issues=(
                    ValidationIssue(
                        reason="missing_symbolic_fixture_inventory",
                        detail="no symbolic publication fixture paths were provided",
                    ),
                ),
            )
        )

    for path in fixture_path_list:
        checks.append(
            validate_symbolic_publication_fixture(
                path,
                live_manifest_builder=live_manifest_builder,
            )
        )

    for path in pdg_path_list:
        checks.append(validate_pdg_payload_file(path))

    if include_default_pdg:
        checks.append(
            validate_pdg_payload(
                _default_pdg_payload(),
                subject="default_pdg_validation_fixture",
            )
        )

    if include_source_execution:
        checks.append(
            validate_source_execution_readiness(
                source_retrieval_run_plan,
                source_max_jobs=source_max_jobs,
                source_job_id=source_job_id,
            )
        )

    report = {
        "report_kind": VALIDATION_REPORT_KIND,
        "ok": all(check.ok for check in checks),
        "summary": {
            "check_count": len(checks),
            "failed_check_count": sum(1 for check in checks if not check.ok),
            "error_count": sum(
                1
                for check in checks
                for issue in check.issues
                if issue.severity == "error"
            ),
        },
        "checks": [check.to_dict() for check in checks],
    }
    _assert_json_serializable(report)
    return report


def validate_source_execution_readiness(
    source_retrieval_run_plan: Any | None = None,
    *,
    source_max_jobs: int | None = None,
    source_job_id: str | Iterable[str] | None = None,
) -> ValidationCheck:
    """Validate that source retrieval steps are executor-ready offline."""

    subject = "source_execution_readiness"
    try:
        plan = source_retrieval_run_plan
        if plan is None:
            plan = build_physics_source_retrieval_run_plan(
                max_jobs=source_max_jobs,
                job_id=source_job_id,
            )
        readiness_report = build_source_execution_readiness_report(plan).to_dict()
    except Exception as exc:
        return ValidationCheck(
            check_id="source_execution_readiness",
            subject=subject,
            issues=(
                ValidationIssue(
                    reason="source_execution_report_build_error",
                    detail=str(exc),
                    subject=subject,
                ),
            ),
        )

    issues = tuple(
        _source_execution_diagnostic_issue(diagnostic)
        for diagnostic in readiness_report.get("diagnostics", ())
        if isinstance(diagnostic, Mapping)
    )
    return ValidationCheck(
        check_id="source_execution_readiness",
        subject=subject,
        issues=issues,
        metadata={
            "report": readiness_report,
            "total_steps": readiness_report["summary"]["total_steps"],
            "diagnostic_count": readiness_report["summary"]["diagnostic_count"],
        },
    )


def validate_symbolic_publication_fixture(
    path: Path | str,
    *,
    live_manifest_builder: Any | None = None,
) -> ValidationCheck:
    """Validate one symbolic publication fixture."""

    fixture_path = Path(path)
    subject = str(fixture_path)
    issues: list[ValidationIssue] = []
    try:
        manifest = json.loads(fixture_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationCheck(
            check_id="symbolic_publication_fixture",
            subject=subject,
            issues=(
                ValidationIssue(
                    reason="fixture_load_error",
                    detail=str(exc),
                    subject=subject,
                ),
            ),
        )

    if not isinstance(manifest, Mapping):
        return ValidationCheck(
            check_id="symbolic_publication_fixture",
            subject=subject,
            issues=(
                ValidationIssue(
                    reason="fixture_not_mapping",
                    detail=f"fixture root is {type(manifest).__name__}",
                    subject=subject,
                ),
            ),
        )

    expressions = _rows(manifest, "artifact_symbolic_expressions")
    variables = _rows(manifest, "artifact_symbolic_variables")
    bounds = _rows(manifest, "artifact_validity_bounds")
    if not expressions:
        issues.append(
            ValidationIssue(
                reason="missing_symbolic_expressions",
                table="artifact_symbolic_expressions",
                subject=subject,
            )
        )
    if not variables:
        issues.append(
            ValidationIssue(
                reason="missing_symbolic_variables",
                table="artifact_symbolic_variables",
                subject=subject,
            )
        )

    issues.extend(_symbolic_expression_standard_issues(expressions, subject=subject))
    issues.extend(_symbolic_variable_standard_issues(variables, subject=subject))
    issues.extend(
        _duplicate_value_issues(expressions, "expression_id", subject=subject)
    )
    issues.extend(_duplicate_value_issues(expressions, "artifact_key", subject=subject))

    bindings = _artifact_bindings_for_manifest(expressions)
    load_result = load_symbolic_publication_manifest(manifest, bindings)
    for diagnostic in load_result.diagnostics:
        issues.append(
            ValidationIssue(
                reason=f"publication_loader_{diagnostic.reason}",
                severity=(
                    diagnostic.severity if diagnostic.severity == "error" else "error"
                ),
                detail=diagnostic.detail,
                table=diagnostic.table,
                subject=diagnostic.artifact_key or subject,
            )
        )

    insert_rows = load_result.to_insert_rows()
    if len(insert_rows["artifact_symbolic_expressions"]) != len(expressions):
        issues.append(
            ValidationIssue(
                reason="symbolic_expression_insert_count_mismatch",
                detail=(
                    f"loaded {len(insert_rows['artifact_symbolic_expressions'])} "
                    f"of {len(expressions)} expression rows"
                ),
                table="artifact_symbolic_expressions",
                subject=subject,
            )
        )
    if len(insert_rows["artifact_symbolic_variables"]) != len(variables):
        issues.append(
            ValidationIssue(
                reason="symbolic_variable_insert_count_mismatch",
                detail=(
                    f"loaded {len(insert_rows['artifact_symbolic_variables'])} "
                    f"of {len(variables)} variable rows"
                ),
                table="artifact_symbolic_variables",
                subject=subject,
            )
        )

    if live_manifest_builder is not None and manifest.get("modules"):
        try:
            live_manifest = live_manifest_builder(modules=tuple(manifest["modules"]))
        except Exception as exc:  # pragma: no cover - defensive CLI boundary
            issues.append(
                ValidationIssue(
                    reason="live_manifest_render_error",
                    detail=str(exc),
                    subject=subject,
                )
            )
        else:
            if _canonical_json(live_manifest) != _canonical_json(manifest):
                issues.append(
                    ValidationIssue(
                        reason="symbolic_fixture_drift",
                        detail="fixture JSON differs from live symbolic manifest render",
                        subject=subject,
                    )
                )

    return ValidationCheck(
        check_id="symbolic_publication_fixture",
        subject=subject,
        issues=tuple(issues),
        metadata={
            "expression_count": len(expressions),
            "variable_count": len(variables),
            "validity_bound_count": len(bounds),
        },
    )


def validate_pdg_payload_file(path: Path | str) -> ValidationCheck:
    """Load and validate one PDG JSON payload file."""

    payload_path = Path(path)
    subject = str(payload_path)
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return ValidationCheck(
            check_id="pdg_publication_graph",
            subject=str(payload_path),
            issues=(
                ValidationIssue(
                    reason="pdg_payload_load_error",
                    detail=str(exc),
                    subject=str(payload_path),
                ),
            ),
        )
    if not isinstance(payload, Mapping):
        return ValidationCheck(
            check_id="pdg_publication_graph",
            subject=subject,
            issues=(
                ValidationIssue(
                    reason="pdg_payload_not_mapping",
                    detail=f"payload root is {type(payload).__name__}",
                    subject=subject,
                ),
            ),
        )
    subject = _pdg_payload_subject(payload, fallback=subject)
    check = validate_pdg_payload(payload, subject=subject)
    return ValidationCheck(
        check_id=check.check_id,
        subject=check.subject,
        issues=check.issues,
        metadata={**dict(check.metadata), "fixture_path": str(payload_path)},
    )


def validate_pdg_payload(
    payload: Mapping[str, Any],
    *,
    subject: str = "pdg_payload",
) -> ValidationCheck:
    """Validate PDG relationship extraction and derived CDG publication rows."""

    issues: list[ValidationIssue] = []
    try:
        bundle = parse_pdg_document(payload)
    except Exception as exc:
        return ValidationCheck(
            check_id="pdg_publication_graph",
            subject=subject,
            issues=(
                ValidationIssue(
                    reason="pdg_parse_error",
                    detail=str(exc),
                    subject=subject,
                ),
            ),
        )

    if not bundle.equations:
        issues.append(ValidationIssue(reason="missing_pdg_equations", subject=subject))
    if not bundle.inference_edges:
        issues.append(
            ValidationIssue(reason="missing_pdg_inference_edges", subject=subject)
        )

    bindings = {
        equation.node_id: _pdg_expression_binding(subject, equation.node_id)
        for equation in bundle.equations
    }
    ingest_result = build_pdg_relationship_ingest(
        bundle,
        expression_bindings_by_pdg_node_id=bindings,
    )
    if ingest_result.skipped_edges:
        for skipped in ingest_result.skipped_edges:
            skipped_reason = _reason_token(str(skipped.get("reason") or "unknown"))
            edge_id = str(skipped.get("pdg_edge_id") or "edge")
            issues.append(
                ValidationIssue(
                    reason=f"pdg_relationship_edge_skipped_{skipped_reason}",
                    detail=_canonical_json(skipped),
                    subject=f"{subject}:{edge_id}",
                )
            )

    publication_rows = build_pdg_publication_write_rows(ingest_result)
    for diagnostic in publication_rows.diagnostics:
        issues.append(
            ValidationIssue(
                reason=f"pdg_publication_{diagnostic.get('reason', 'diagnostic')}",
                severity=str(diagnostic.get("severity") or "error"),
                detail=str(diagnostic.get("detail") or ""),
                table=str(diagnostic.get("table") or ""),
                subject=subject,
            )
        )
    for diagnostic in validate_pdg_cdg_publication_graph(publication_rows):
        issues.append(
            ValidationIssue(
                reason=f"pdg_cdg_graph_{diagnostic.get('reason', 'diagnostic')}",
                severity=str(diagnostic.get("severity") or "error"),
                detail=str(diagnostic.get("detail") or ""),
                table=str(diagnostic.get("table") or ""),
                subject=subject,
            )
        )

    second_rows = build_pdg_publication_write_rows(
        build_pdg_relationship_ingest(
            bundle,
            expression_bindings_by_pdg_node_id=bindings,
        )
    )
    if publication_rows.to_insert_rows() != second_rows.to_insert_rows():
        issues.append(
            ValidationIssue(
                reason="pdg_publication_rows_nondeterministic",
                subject=subject,
            )
        )

    insert_rows = publication_rows.to_insert_rows()
    return ValidationCheck(
        check_id="pdg_publication_graph",
        subject=subject,
        issues=tuple(issues),
        metadata={
            "equation_count": len(bundle.equations),
            "inference_edge_count": len(bundle.inference_edges),
            "relationship_row_count": len(
                insert_rows.get("artifact_relationships", ())
            ),
            "cdg_node_count": len(insert_rows.get("artifact_cdg_nodes", ())),
            "cdg_edge_count": len(insert_rows.get("artifact_cdg_edges", ())),
            "cdg_binding_count": len(insert_rows.get("artifact_cdg_bindings", ())),
        },
    )


def discover_symbolic_fixture_paths(atoms_repo: Path | str) -> tuple[Path, ...]:
    """Return checked-in publication manifest fixtures from an atoms repo."""

    root = Path(atoms_repo)
    fixture_dir = root / "data" / "publication_fixtures"
    return tuple(sorted(fixture_dir.glob("*.publication_manifest.json")))


def discover_pdg_payload_fixture_paths(
    root: Path | str | None = None,
) -> tuple[Path, ...]:
    """Return checked-in local PDG payload fixtures for offline validation."""

    repo_root = Path(root) if root is not None else _REPO_ROOT
    paths: list[Path] = []
    for pattern in _DEFAULT_PDG_FIXTURE_GLOBS:
        paths.extend(repo_root.glob(str(pattern)))
    return tuple(sorted(path for path in paths if path.is_file()))


def _symbolic_expression_standard_issues(
    expressions: Sequence[Mapping[str, Any]],
    *,
    subject: str,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    for index, row in enumerate(expressions):
        row_subject = _row_subject(row, subject=subject, index=index)
        for field_name in (
            "expression_id",
            "artifact_key",
            "atom_name",
            "raw_formula",
            "sympy_srepr",
            "canonical_expr_hash",
            "topology_hash",
            "dimensional_hash",
        ):
            if not row.get(field_name):
                issues.append(
                    ValidationIssue(
                        reason=f"missing_{field_name}",
                        table="artifact_symbolic_expressions",
                        subject=row_subject,
                    )
                )
        if not _string_sequence(row.get("mechanism_tags")):
            issues.append(
                ValidationIssue(
                    reason="missing_mechanism_tags",
                    table="artifact_symbolic_expressions",
                    subject=row_subject,
                )
            )
        if not _string_sequence(row.get("behavioral_archetypes")):
            issues.append(
                ValidationIssue(
                    reason="missing_behavioral_archetypes",
                    table="artifact_symbolic_expressions",
                    subject=row_subject,
                )
            )
        if not _string_sequence(row.get("bibliography")):
            issues.append(
                ValidationIssue(
                    reason="missing_bibliography",
                    table="artifact_symbolic_expressions",
                    subject=row_subject,
                )
            )
        if row.get("review_status") not in {"automated_pass", "human_reviewed"}:
            issues.append(
                ValidationIssue(
                    reason="review_status_not_publishable",
                    detail=str(row.get("review_status") or ""),
                    table="artifact_symbolic_expressions",
                    subject=row_subject,
                )
            )
        if row.get("validation_status") != "passed":
            issues.append(
                ValidationIssue(
                    reason="validation_status_not_passed",
                    detail=str(row.get("validation_status") or ""),
                    table="artifact_symbolic_expressions",
                    subject=row_subject,
                )
            )
    return tuple(issues)


def _symbolic_variable_standard_issues(
    variables: Sequence[Mapping[str, Any]],
    *,
    subject: str,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    for index, row in enumerate(variables):
        row_subject = _row_subject(row, subject=subject, index=index)
        if not row.get("symbol_name") and not row.get("symbol"):
            issues.append(
                ValidationIssue(
                    reason="missing_symbol_name",
                    table="artifact_symbolic_variables",
                    subject=row_subject,
                )
            )
        if not row.get("dim_signature"):
            issues.append(
                ValidationIssue(
                    reason="missing_dim_signature",
                    table="artifact_symbolic_variables",
                    subject=row_subject,
                )
            )
    return tuple(issues)


def _duplicate_value_issues(
    rows: Sequence[Mapping[str, Any]],
    field_name: str,
    *,
    subject: str,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        value = str(row.get(field_name) or "")
        if not value:
            continue
        first_index = seen.get(value)
        if first_index is None:
            seen[value] = index
            continue
        issues.append(
            ValidationIssue(
                reason=f"duplicate_{field_name}",
                detail=_canonical_json(
                    {
                        "value": value,
                        "first_row_index": first_index,
                        "row_index": index,
                    }
                ),
                table="artifact_symbolic_expressions",
                subject=subject,
            )
        )
    return tuple(issues)


def _source_execution_diagnostic_issue(
    diagnostic: Mapping[str, Any],
) -> ValidationIssue:
    code = _reason_token(str(diagnostic.get("code") or "diagnostic"))
    subject_parts = [
        "source_execution_readiness",
        str(diagnostic.get("job_id") or ""),
        str(diagnostic.get("step_index") or ""),
    ]
    return ValidationIssue(
        reason=f"source_execution_{code}",
        severity=str(diagnostic.get("severity") or "error"),
        detail=str(diagnostic.get("message") or ""),
        table="source_execution_readiness",
        subject=":".join(part for part in subject_parts if part),
    )


def _artifact_bindings_for_manifest(
    expressions: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for row in expressions:
        key = str(
            row.get("local_artifact_key")
            or row.get("artifact_key")
            or row.get("atom_name")
            or ""
        )
        if not key:
            continue
        bindings[key] = {
            "artifact_id": str(uuid5(_ARTIFACT_NAMESPACE, key)),
            "version_id": str(uuid5(_VERSION_NAMESPACE, key)),
        }
    return bindings


def _pdg_expression_binding(subject: str, node_id: str) -> dict[str, Any]:
    expression_id = str(uuid5(_EXPRESSION_NAMESPACE, f"{subject}:{node_id}"))
    token = _fqdn_token(f"{subject}:{node_id}")
    content_hash = uuid5(_VERSION_NAMESPACE, expression_id).hex
    return {
        "expression_id": expression_id,
        "metadata": {
            "bound_artifact_fqdn": f"physics.validation.{token}",
            "bound_version_content_hash": content_hash,
            "binding_confidence": 1.0,
            "binding_source": "offline_validation",
        },
    }


def _pdg_payload_subject(payload: Mapping[str, Any], *, fallback: str) -> str:
    for container in (
        payload,
        _mapping_value(payload.get("metadata")),
        _mapping_value(payload.get("fixture")),
    ):
        subject = _first_string(
            container,
            "validation_subject",
            "fixture_subject",
            "subject",
        )
        if subject:
            return subject
        fixture_id = _first_string(container, "fixture_id", "id", "name")
        if fixture_id:
            return f"pdg_fixture:{_reason_token(fixture_id)}"
    return fallback


def _mapping_value(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _first_string(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _load_live_manifest_builder(atoms_repo: Path | str | None) -> Any | None:
    if atoms_repo is None:
        return None
    src = Path(atoms_repo) / "src"
    if not src.exists():
        return None
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    try:
        from sciona.atoms.physics.symbolic_publication_manifest import (  # type: ignore
            build_symbolic_publication_manifest,
        )
    except Exception:
        return None
    return build_symbolic_publication_manifest


def _rows(manifest: Mapping[str, Any], key: str) -> tuple[Mapping[str, Any], ...]:
    value = manifest.get(key)
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(row for row in value if isinstance(row, Mapping))


def _string_sequence(value: Any) -> tuple[str, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return ()
    return tuple(str(item) for item in value if str(item))


def _row_subject(row: Mapping[str, Any], *, subject: str, index: int) -> str:
    return str(row.get("artifact_key") or row.get("atom_name") or f"{subject}#{index}")


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _assert_json_serializable(report: Mapping[str, Any]) -> None:
    try:
        json.dumps(report, sort_keys=True, ensure_ascii=True)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise ValueError("validation report must be JSON serializable") from exc


def _fqdn_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() else "_" for character in value.lower()
    )
    return "_".join(part for part in token.split("_") if part) or "pdg"


def _reason_token(value: str) -> str:
    token = "".join(
        character if character.isalnum() else "_" for character in value.lower()
    )
    return "_".join(part for part in token.split("_") if part) or "unknown"


def _default_pdg_payload() -> dict[str, Any]:
    return {
        "equations": [
            {
                "id": "eq:base",
                "label": "Newton's second law",
                "latex": "F = m a",
            },
            {
                "id": "eq:solved",
                "label": "Acceleration from force",
                "latex": "a = F / m",
            },
            {
                "id": "eq:force",
                "label": "Constant mass force",
                "latex": "F(t) = m d^2x/dt^2",
            },
        ],
        "inference_edges": [
            {
                "id": "edge:solve",
                "source": "eq:base",
                "target": "eq:solved",
                "rule": "solve for acceleration",
                "confidence": 0.93,
                "bindings": {"solve_for": "a"},
            },
            {
                "id": "edge:substitute",
                "source": "eq:solved",
                "target": "eq:force",
                "rule": "substitution",
                "assumptions": ["mass is constant"],
                "confidence": 0.81,
            },
        ],
    }


__all__ = [
    "VALIDATION_REPORT_KIND",
    "ValidationCheck",
    "ValidationIssue",
    "build_physics_ingestion_validation_report",
    "discover_pdg_payload_fixture_paths",
    "discover_symbolic_fixture_paths",
    "validate_pdg_payload",
    "validate_pdg_payload_file",
    "validate_source_execution_readiness",
    "validate_symbolic_publication_fixture",
]

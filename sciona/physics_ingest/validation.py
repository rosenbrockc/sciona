"""Offline validation for physics symbolic fixtures and PDG CDG rows."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
import fnmatch
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from sciona.physics_ingest.pdg_cdg import (
    PDGCDGArtifactEnvelope,
    build_pdg_publication_write_rows,
    build_pdg_relationship_ingest,
    validate_pdg_cdg_publication_graph,
)
from sciona.physics_ingest.publication import load_symbolic_publication_manifest
from sciona.physics_ingest.sources import (
    build_physics_source_retrieval_run_plan,
    build_source_adapter_coverage_report,
    build_source_execution_readiness_report,
)
from sciona.physics_ingest.sources.pdg import parse_pdg_document


VALIDATION_REPORT_KIND = "physics_ingestion_validation"
VALIDATION_REPORT_VERSION = "physics-ingestion-validation.v1"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_PDG_FIXTURE_GLOBS = (
    Path("tests") / "physics_ingest" / "fixtures" / "pdg_payloads" / "*.pdg.json",
    Path("docs") / "physics_ingest" / "fixtures" / "pdg_payloads" / "*.pdg.json",
)
_SYMBOLIC_FIXTURE_DIR = Path("data") / "publication_fixtures"
_SYMBOLIC_FIXTURE_SUFFIX = ".publication_manifest.json"
_ARTIFACT_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.validation.artifact")
_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.validation.version")
_EXPRESSION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.validation.expression",
)
_DEFAULT_PDG_CDG_ARTIFACT_ENVELOPE = PDGCDGArtifactEnvelope(
    fqdn_prefix="physics.validation.pdg.cdg",
    semver="0.1.0",
    namespace_root="physics",
    namespace_path="validation/pdg/cdg",
    source_package="pdg",
    source_module_path="pdg.validation",
    source_symbol_prefix="pdg_cdg_validation_candidate",
    status="draft",
    visibility_tier="internal",
    description="Offline validation PDG-derived CDG manifest",
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
    include_source_adapter_coverage: bool = True,
    include_source_adapter_data_artifact_seeds: bool = True,
    source_retrieval_run_plan: Any | None = None,
    source_retrieval_manifest: Any | None = None,
    source_max_jobs: int | None = None,
    source_job_id: str | Iterable[str] | None = None,
    source_phase7_ring: str | Iterable[str] | None = None,
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
    if strict and include_default_pdg and not pdg_path_list:
        checks.append(
            ValidationCheck(
                check_id="pdg_payload_fixture_inventory",
                subject="pdg_payload_fixtures",
                issues=(
                    ValidationIssue(
                        reason="missing_pdg_payload_fixture_inventory",
                        detail="no local PDG payload fixture paths were provided",
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
                source_phase7_ring=source_phase7_ring,
            )
        )

    if include_source_adapter_coverage:
        checks.append(validate_source_adapter_coverage(source_retrieval_manifest))

    if include_source_adapter_data_artifact_seeds:
        checks.append(validate_source_adapter_data_artifact_seed_quality())

    report = {
        "report_kind": VALIDATION_REPORT_KIND,
        "report_version": VALIDATION_REPORT_VERSION,
        "ok": all(check.ok for check in checks),
        "summary": _validation_report_summary(checks),
        "checks": [check.to_dict() for check in checks],
    }
    _assert_json_serializable(report)
    return report


def _validation_report_summary(checks: Sequence[ValidationCheck]) -> dict[str, Any]:
    """Build stable aggregate fields for CI and dashboard consumers."""

    failed_checks = tuple(check for check in checks if not check.ok)
    errors = tuple(
        issue
        for check in checks
        for issue in check.issues
        if issue.severity == "error"
    )
    issues = tuple(issue for check in checks for issue in check.issues)

    return {
        "check_count": len(checks),
        "failed_check_count": len(failed_checks),
        "error_count": len(errors),
        "check_ids": [check.check_id for check in checks],
        "failed_check_ids": [check.check_id for check in failed_checks],
        "error_count_by_reason": _sorted_count_dict(issue.reason for issue in errors),
        "failed_count_by_check_id": _sorted_count_dict(
            check.check_id for check in failed_checks
        ),
        "issue_count_by_table": _sorted_count_dict(
            issue.table or "unscoped" for issue in issues
        ),
    }


def _sorted_count_dict(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {key: counts[key] for key in sorted(counts)}


def validate_source_adapter_coverage(
    source_retrieval_manifest: Any | None = None,
) -> ValidationCheck:
    """Validate that retrieval jobs have covered offline source adapters."""

    subject = "source_adapter_coverage"
    try:
        coverage_report = build_source_adapter_coverage_report(
            source_retrieval_manifest
        ).to_dict()
    except Exception as exc:
        return ValidationCheck(
            check_id="source_adapter_coverage",
            subject=subject,
            issues=(
                ValidationIssue(
                    reason="source_adapter_coverage_report_build_error",
                    detail=str(exc),
                    subject=subject,
                ),
            ),
        )

    issues = tuple(
        _source_adapter_coverage_diagnostic_issue(diagnostic)
        for diagnostic in coverage_report.get("diagnostics", ())
        if isinstance(diagnostic, Mapping)
    )
    summary = coverage_report["summary"]
    return ValidationCheck(
        check_id="source_adapter_coverage",
        subject=subject,
        issues=issues,
        metadata={
            "report": coverage_report,
            "total_jobs": summary["total_jobs"],
            "covered": summary["covered"],
            "blocked": summary["blocked"],
            "diagnostic_count": summary["diagnostic_count"],
        },
    )


def validate_source_adapter_data_artifact_seed_quality(
    adapter_bundles: Mapping[str, Any] | Iterable[tuple[str, Any]] | None = None,
) -> ValidationCheck:
    """Validate offline source adapter future data-artifact seed dictionaries."""

    subject = "source_adapter_data_artifact_seeds"
    issues: list[ValidationIssue] = []
    bundle_summaries: list[dict[str, Any]] = []
    seen_fqdns: dict[str, str] = {}
    seen_source_identities: dict[tuple[str, str], str] = {}
    seed_count = 0

    if adapter_bundles is None:
        bundle_items = _build_default_data_artifact_seed_validation_bundles(issues)
    elif isinstance(adapter_bundles, Mapping):
        bundle_items = tuple(
            (str(bundle_id), bundle) for bundle_id, bundle in adapter_bundles.items()
        )
    else:
        bundle_items = tuple(
            (str(bundle_id), bundle) for bundle_id, bundle in adapter_bundles
        )

    for bundle_id, bundle in bundle_items:
        seeds = _bundle_data_artifact_seeds(bundle)
        bundle_summaries.append(
            {
                "bundle_id": bundle_id,
                "seed_count": len(seeds),
            }
        )
        if not seeds:
            issues.append(
                ValidationIssue(
                    reason="source_adapter_data_artifact_seed_missing_inventory",
                    detail="adapter bundle emitted no data_artifact_seeds",
                    table=subject,
                    subject=f"{subject}:{bundle_id}",
                )
            )
            continue
        for index, seed in enumerate(seeds):
            seed_count += 1
            if not isinstance(seed, Mapping):
                issues.append(
                    ValidationIssue(
                        reason="source_adapter_data_artifact_seed_not_mapping",
                        detail=f"seed root is {type(seed).__name__}",
                        table=subject,
                        subject=f"{subject}:{bundle_id}:{index}",
                    )
                )
                continue
            issues.extend(
                _data_artifact_seed_issues(seed, bundle_id=bundle_id, index=index)
            )
            issues.extend(
                _duplicate_data_artifact_seed_issues(
                    seed,
                    bundle_id=bundle_id,
                    index=index,
                    seen_fqdns=seen_fqdns,
                    seen_source_identities=seen_source_identities,
                )
            )

    report = {
        "summary": {
            "bundle_count": len(bundle_items),
            "seed_count": seed_count,
            "diagnostic_count": len(issues),
        },
        "bundles": bundle_summaries,
    }
    return ValidationCheck(
        check_id="source_adapter_data_artifact_seeds",
        subject=subject,
        issues=tuple(issues),
        metadata={
            "report": report,
            "bundle_count": report["summary"]["bundle_count"],
            "seed_count": report["summary"]["seed_count"],
            "diagnostic_count": report["summary"]["diagnostic_count"],
        },
    )


def validate_source_execution_readiness(
    source_retrieval_run_plan: Any | None = None,
    *,
    source_max_jobs: int | None = None,
    source_job_id: str | Iterable[str] | None = None,
    source_phase7_ring: str | Iterable[str] | None = None,
) -> ValidationCheck:
    """Validate that source retrieval steps are executor-ready offline."""

    subject = "source_execution_readiness"
    try:
        plan = source_retrieval_run_plan
        if plan is None:
            plan = build_physics_source_retrieval_run_plan(
                max_jobs=source_max_jobs,
                job_id=source_job_id,
                phase7_ring=source_phase7_ring,
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
    issues.extend(_symbolic_variable_duplicate_issues(variables, subject=subject))
    issues.extend(
        _validity_bound_standard_issues(
            bounds,
            expressions=expressions,
            variables=variables,
            subject=subject,
        )
    )
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
    cdg_artifact_envelope: PDGCDGArtifactEnvelope | Mapping[str, Any] | None = (
        _DEFAULT_PDG_CDG_ARTIFACT_ENVELOPE
    ),
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

    try:
        publication_rows = build_pdg_publication_write_rows(
            ingest_result,
            cdg_artifact_envelope=cdg_artifact_envelope,
        )
    except Exception as exc:
        issues.append(
            ValidationIssue(
                reason="pdg_cdg_artifact_envelope_error",
                detail=str(exc),
                subject=subject,
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

    insert_rows = publication_rows.to_insert_rows()
    cdg_candidate_manifest_count = len(ingest_result.cdg_candidate_manifests)
    artifact_row_count = len(insert_rows.get("artifacts", ()))
    artifact_version_row_count = len(insert_rows.get("artifact_versions", ()))
    if cdg_candidate_manifest_count:
        if artifact_row_count != cdg_candidate_manifest_count:
            issues.append(
                ValidationIssue(
                    reason="pdg_cdg_artifact_envelope_artifacts_missing",
                    detail=_canonical_json(
                        {
                            "cdg_candidate_manifest_count": cdg_candidate_manifest_count,
                            "artifact_row_count": artifact_row_count,
                        }
                    ),
                    table="artifacts",
                    subject=subject,
                )
            )
        if artifact_version_row_count != cdg_candidate_manifest_count:
            issues.append(
                ValidationIssue(
                    reason="pdg_cdg_artifact_envelope_artifact_versions_missing",
                    detail=_canonical_json(
                        {
                            "cdg_candidate_manifest_count": cdg_candidate_manifest_count,
                            "artifact_version_row_count": artifact_version_row_count,
                        }
                    ),
                    table="artifact_versions",
                    subject=subject,
                )
            )

    try:
        second_rows = build_pdg_publication_write_rows(
            build_pdg_relationship_ingest(
                bundle,
                expression_bindings_by_pdg_node_id=bindings,
            ),
            cdg_artifact_envelope=cdg_artifact_envelope,
        )
    except Exception as exc:
        issues.append(
            ValidationIssue(
                reason="pdg_cdg_artifact_envelope_error",
                detail=str(exc),
                subject=subject,
            )
        )
        second_rows = build_pdg_publication_write_rows(
            build_pdg_relationship_ingest(
                bundle,
                expression_bindings_by_pdg_node_id=bindings,
            )
        )
    second_insert_rows = second_rows.to_insert_rows()
    if publication_rows.to_insert_rows() != second_insert_rows:
        issues.append(
            ValidationIssue(
                reason="pdg_publication_rows_nondeterministic",
                subject=subject,
            )
        )
    if cdg_candidate_manifest_count and (
        insert_rows.get("artifacts", ()) != second_insert_rows.get("artifacts", ())
        or insert_rows.get("artifact_versions", ())
        != second_insert_rows.get("artifact_versions", ())
    ):
        issues.append(
            ValidationIssue(
                reason="pdg_cdg_artifact_envelope_rows_nondeterministic",
                table="artifacts",
                subject=subject,
            )
        )

    return ValidationCheck(
        check_id="pdg_publication_graph",
        subject=subject,
        issues=tuple(issues),
        metadata={
            "equation_count": len(bundle.equations),
            "inference_edge_count": len(bundle.inference_edges),
            "cdg_candidate_manifest_count": cdg_candidate_manifest_count,
            "artifact_row_count": artifact_row_count,
            "artifact_version_row_count": artifact_version_row_count,
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
    fixture_dir = root / _SYMBOLIC_FIXTURE_DIR
    return tuple(sorted(fixture_dir.glob(f"*{_SYMBOLIC_FIXTURE_SUFFIX}")))


def discover_changed_symbolic_fixture_paths(atoms_repo: Path | str) -> tuple[Path, ...]:
    """Return git-changed symbolic publication fixtures from an atoms repo."""

    root = Path(atoms_repo)
    return tuple(
        path
        for path in discover_git_changed_paths(root)
        if _is_symbolic_fixture_path(path, root)
    )


def discover_pdg_payload_fixture_paths(
    root: Path | str | None = None,
) -> tuple[Path, ...]:
    """Return checked-in local PDG payload fixtures for offline validation."""

    repo_root = Path(root) if root is not None else _REPO_ROOT
    paths: list[Path] = []
    for pattern in _DEFAULT_PDG_FIXTURE_GLOBS:
        paths.extend(repo_root.glob(str(pattern)))
    return tuple(sorted(path for path in paths if path.is_file()))


def discover_changed_pdg_payload_fixture_paths(
    root: Path | str | None = None,
) -> tuple[Path, ...]:
    """Return git-changed local PDG payload fixtures for offline validation.

    Changed-only discovery intentionally follows the normal PDG fixture globs and
    the ``*.pdg.json`` suffix. Intentionally invalid neighboring ``*.json`` files
    stay out of the developer-loop mode unless passed explicitly with
    ``--pdg-json``.
    """

    repo_root = Path(root) if root is not None else _REPO_ROOT
    return tuple(
        path
        for path in discover_git_changed_paths(repo_root)
        if _is_pdg_payload_fixture_path(path, repo_root)
    )


def discover_git_changed_paths(root: Path | str) -> tuple[Path, ...]:
    """Return existing files changed in a local git worktree.

    The helper is deterministic and offline: it combines local unstaged changes,
    staged changes, and untracked files without requiring a clean working tree.
    Deleted paths are omitted because discovered validation inputs must be
    readable; explicit CLI paths are still validated by the normal loaders.
    """

    repo_root = Path(root)
    if not repo_root.exists():
        return ()

    relative_paths: set[Path] = set()
    for git_args in (
        ("diff", "--name-only", "--diff-filter=ACMR"),
        ("diff", "--cached", "--name-only", "--diff-filter=ACMR"),
        ("ls-files", "--others", "--exclude-standard"),
    ):
        try:
            completed = subprocess.run(
                ("git", "-C", str(repo_root), *git_args),
                check=True,
                capture_output=True,
                text=True,
            )
        except (OSError, subprocess.CalledProcessError):
            return ()
        for line in completed.stdout.splitlines():
            if line:
                relative_paths.add(Path(line))

    return tuple(
        sorted(
            (
                repo_root / relative_path
                for relative_path in relative_paths
                if (repo_root / relative_path).is_file()
            ),
            key=lambda path: path.as_posix(),
        )
    )


def _is_symbolic_fixture_path(path: Path, root: Path) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return False
    return (
        relative_path.parent == _SYMBOLIC_FIXTURE_DIR
        and relative_path.name.endswith(_SYMBOLIC_FIXTURE_SUFFIX)
    )


def _is_pdg_payload_fixture_path(path: Path, root: Path) -> bool:
    try:
        relative_path = path.relative_to(root)
    except ValueError:
        return False
    relative_posix = relative_path.as_posix()
    return any(
        fnmatch.fnmatchcase(relative_posix, pattern.as_posix())
        for pattern in _DEFAULT_PDG_FIXTURE_GLOBS
    )


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


def _symbolic_variable_duplicate_issues(
    variables: Sequence[Mapping[str, Any]],
    *,
    subject: str,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    seen_references: dict[tuple[str, str, str], int] = {}
    seen_source_ids: dict[tuple[str, str, str], int] = {}
    for index, row in enumerate(variables):
        group_key = _symbolic_variable_group_key(row, fallback=subject)
        row_subject = _row_subject(row, subject=subject, index=index)

        variable_reference = _text_value(
            row,
            "symbol_name",
            "symbol",
            "source_symbol",
            "variable_name",
        )
        if variable_reference:
            issue = _duplicate_symbolic_variable_issue(
                seen_references,
                key=(*group_key, variable_reference),
                value_field="variable_reference",
                value=variable_reference,
                reason="duplicate_symbolic_variable_reference",
                row_index=index,
                row_subject=row_subject,
            )
            if issue is not None:
                issues.append(issue)

        source_variable_id = _text_value(row, "source_variable_id")
        if source_variable_id:
            issue = _duplicate_symbolic_variable_issue(
                seen_source_ids,
                key=(*group_key, source_variable_id),
                value_field="source_variable_id",
                value=source_variable_id,
                reason="duplicate_source_variable_id",
                row_index=index,
                row_subject=row_subject,
            )
            if issue is not None:
                issues.append(issue)
    return tuple(issues)


def _symbolic_variable_group_key(
    row: Mapping[str, Any],
    *,
    fallback: str,
) -> tuple[str, str]:
    return (
        _text_value(row, "expression_id") or fallback,
        _text_value(row, "local_artifact_key", "artifact_key", "atom_name") or fallback,
    )


def _duplicate_symbolic_variable_issue(
    seen: dict[tuple[str, str, str], int],
    *,
    key: tuple[str, str, str],
    value_field: str,
    value: str,
    reason: str,
    row_index: int,
    row_subject: str,
) -> ValidationIssue | None:
    first_index = seen.get(key)
    if first_index is None:
        seen[key] = row_index
        return None
    expression_id, artifact_key, _ = key
    return ValidationIssue(
        reason=reason,
        detail=_canonical_json(
            {
                "artifact_key": artifact_key,
                "expression_id": expression_id,
                "first_row_index": first_index,
                "row_index": row_index,
                value_field: value,
            }
        ),
        table="artifact_symbolic_variables",
        subject=row_subject,
    )


def _validity_bound_standard_issues(
    bounds: Sequence[Mapping[str, Any]],
    *,
    expressions: Sequence[Mapping[str, Any]],
    variables: Sequence[Mapping[str, Any]],
    subject: str,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    expression_ids = {
        _text_value(row, "expression_id")
        for row in expressions
        if row.get("expression_id")
    }
    expression_keys = {
        key for row in expressions for key in _publication_candidate_keys(row)
    }
    variable_keys = {
        (expression_key, symbol)
        for row in variables
        for expression_key in _publication_candidate_keys(row)
        for symbol in _symbol_reference_values(row)
    }

    for index, row in enumerate(bounds):
        row_subject = _row_subject(row, subject=subject, index=index)
        bound_expression_id = _text_value(row, "expression_id")
        bound_expression_keys = tuple(
            key for key in _publication_candidate_keys(row) if key in expression_keys
        )
        if bound_expression_id and bound_expression_id not in expression_ids:
            issues.append(
                ValidationIssue(
                    reason="validity_bound_expression_id_unknown",
                    detail=_canonical_json({"expression_id": bound_expression_id}),
                    table="artifact_validity_bounds",
                    subject=row_subject,
                )
            )
        if not bound_expression_keys:
            issues.append(
                ValidationIssue(
                    reason="validity_bound_missing_expression_reference",
                    detail=_canonical_json(
                        {"candidate_keys": _publication_candidate_keys(row)}
                    ),
                    table="artifact_validity_bounds",
                    subject=row_subject,
                )
            )

        if row.get("review_status") not in {"automated_pass", "human_reviewed"}:
            issues.append(
                ValidationIssue(
                    reason="validity_bound_review_status_not_publishable",
                    detail=str(row.get("review_status") or ""),
                    table="artifact_validity_bounds",
                    subject=row_subject,
                )
            )

        lower = row.get("lower_value", row.get("min_value"))
        upper = row.get("upper_value", row.get("max_value"))
        if _real_number(lower) and _real_number(upper) and lower > upper:
            issues.append(
                ValidationIssue(
                    reason="validity_bound_range_inverted",
                    detail=_canonical_json(
                        {"lower_value": lower, "upper_value": upper}
                    ),
                    table="artifact_validity_bounds",
                    subject=row_subject,
                )
            )

        scope = _text_value(row, "scope") or "expression"
        symbol_values = _symbol_reference_values(row)
        if scope == "variable" and not symbol_values:
            issues.append(
                ValidationIssue(
                    reason="validity_bound_missing_variable_reference",
                    table="artifact_validity_bounds",
                    subject=row_subject,
                )
            )
            continue
        if symbol_values and bound_expression_keys:
            if not any(
                (expression_key, symbol) in variable_keys
                for expression_key in bound_expression_keys
                for symbol in symbol_values
            ):
                issues.append(
                    ValidationIssue(
                        reason="validity_bound_variable_not_found",
                        detail=_canonical_json(
                            {
                                "candidate_expression_keys": bound_expression_keys,
                                "candidate_symbols": symbol_values,
                            }
                        ),
                        table="artifact_validity_bounds",
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


def _source_adapter_coverage_diagnostic_issue(
    diagnostic: Mapping[str, Any],
) -> ValidationIssue:
    code = _reason_token(str(diagnostic.get("code") or "diagnostic"))
    subject_parts = [
        "source_adapter_coverage",
        str(diagnostic.get("job_id") or ""),
    ]
    return ValidationIssue(
        reason=f"source_adapter_coverage_{code}",
        severity=str(diagnostic.get("severity") or "error"),
        detail=str(diagnostic.get("message") or ""),
        table="source_adapter_coverage",
        subject=":".join(part for part in subject_parts if part),
    )


def _data_artifact_seed_issues(
    seed: Mapping[str, Any],
    *,
    bundle_id: str,
    index: int,
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    row_subject = _data_artifact_seed_subject(seed, bundle_id=bundle_id, index=index)
    for field_name in ("artifact_kind", "fqdn", "source_system", "source_id"):
        value = seed.get(field_name)
        if not isinstance(value, str) or not value.strip():
            issues.append(
                ValidationIssue(
                    reason=f"source_adapter_data_artifact_seed_missing_{field_name}",
                    table="source_adapter_data_artifact_seeds",
                    subject=row_subject,
                )
            )

    try:
        json.dumps(seed, sort_keys=True, ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        issues.append(
            ValidationIssue(
                reason="source_adapter_data_artifact_seed_json_unsafe",
                detail=str(exc),
                table="source_adapter_data_artifact_seeds",
                subject=row_subject,
            )
        )
    return tuple(issues)


def _data_artifact_seed_subject(
    seed: Mapping[str, Any],
    *,
    bundle_id: str,
    index: int,
) -> str:
    identity = str(seed.get("fqdn") or seed.get("source_id") or index)
    return f"source_adapter_data_artifact_seeds:{bundle_id}:{identity}"


def _duplicate_data_artifact_seed_issues(
    seed: Mapping[str, Any],
    *,
    bundle_id: str,
    index: int,
    seen_fqdns: dict[str, str],
    seen_source_identities: dict[tuple[str, str], str],
) -> tuple[ValidationIssue, ...]:
    issues: list[ValidationIssue] = []
    row_subject = _data_artifact_seed_subject(seed, bundle_id=bundle_id, index=index)

    fqdn = seed.get("fqdn")
    if isinstance(fqdn, str) and fqdn.strip():
        fqdn_key = fqdn.strip()
        first_subject = seen_fqdns.setdefault(fqdn_key, row_subject)
        if first_subject != row_subject:
            issues.append(
                ValidationIssue(
                    reason="source_adapter_data_artifact_seed_duplicate_fqdn",
                    detail=f"duplicates {first_subject}",
                    table="source_adapter_data_artifact_seeds",
                    subject=row_subject,
                )
            )

    source_system = seed.get("source_system")
    source_id = seed.get("source_id")
    if (
        isinstance(source_system, str)
        and source_system.strip()
        and isinstance(source_id, str)
        and source_id.strip()
    ):
        identity_key = (source_system.strip(), source_id.strip())
        first_subject = seen_source_identities.setdefault(identity_key, row_subject)
        if first_subject != row_subject:
            issues.append(
                ValidationIssue(
                    reason="source_adapter_data_artifact_seed_duplicate_source_identity",
                    detail=f"duplicates {first_subject}",
                    table="source_adapter_data_artifact_seeds",
                    subject=row_subject,
                )
            )

    return tuple(issues)


def _bundle_data_artifact_seeds(bundle: Any) -> tuple[Any, ...]:
    if isinstance(bundle, Mapping):
        seeds = bundle.get("data_artifact_seeds") or ()
    else:
        seeds = getattr(bundle, "data_artifact_seeds", ()) or ()
    if isinstance(seeds, Sequence) and not isinstance(seeds, (str, bytes, bytearray)):
        return tuple(seeds)
    return (seeds,)


def _build_default_data_artifact_seed_validation_bundles(
    issues: list[ValidationIssue],
) -> tuple[tuple[str, Any], ...]:
    bundles: list[tuple[str, Any]] = []
    for bundle_id, builder in _default_data_artifact_seed_validation_builders():
        try:
            bundles.append((bundle_id, builder()))
        except Exception as exc:  # pragma: no cover - defensive import boundary
            issues.append(
                ValidationIssue(
                    reason="source_adapter_data_artifact_seed_bundle_build_error",
                    detail=f"{type(exc).__name__}: {exc}",
                    table="source_adapter_data_artifact_seeds",
                    subject=f"source_adapter_data_artifact_seeds:{bundle_id}",
                )
            )
    return tuple(bundles)


def _default_data_artifact_seed_validation_builders(
) -> tuple[tuple[str, Any], ...]:
    return (
        ("hitran_lines.backfill", _build_hitran_validation_bundle),
        (
            "materials_project_documents.backfill",
            _build_materials_project_validation_bundle,
        ),
        ("nist_codata_constants.backfill", _build_nist_codata_validation_bundle),
        ("opb_problem_payloads.backfill", _build_opb_validation_bundle),
        ("phy_srbench_payloads.backfill", _build_phy_srbench_validation_bundle),
        ("theoria_payloads.backfill", _build_theoria_validation_bundle),
    )


def _build_hitran_validation_bundle() -> Any:
    from sciona.physics_ingest.sources.hitran import build_hitran_wave0_bundle

    return build_hitran_wave0_bundle(
        (
            {
                "id": "HITRAN:H2O:validation",
                "molecule": "H2O",
                "isotopologue": "1H2-16O",
                "transition": "000-000 P(3)",
                "nu": "1234.56789",
            },
        ),
        source_version="offline validation fixture",
        source_uri="https://example.invalid/hitran-validation",
        retrieved_at="2026-04-30T00:00:00Z",
    )


def _build_materials_project_validation_bundle() -> Any:
    from sciona.physics_ingest.sources.materials_project import (
        build_materials_project_wave0_bundle,
    )

    return build_materials_project_wave0_bundle(
        (
            {
                "material_id": "mp-validation",
                "formula_pretty": "Si",
                "band_gap": 1.17,
            },
        ),
        source_version="offline validation fixture",
        source_uri="https://example.invalid/materials-project-validation",
        retrieved_at="2026-04-30T00:00:00Z",
    )


def _build_nist_codata_validation_bundle() -> Any:
    from sciona.physics_ingest.sources.nist import (
        build_codata_wave0_bundle,
        parse_codata_ascii,
    )

    constants = parse_codata_ascii(
        "speed of light in vacuum | 299 792 458 | (exact) | m s^-1 | symbol=c",
        source_version="offline validation fixture",
        reference_ids=("NIST-CODATA-validation",),
    )
    return build_codata_wave0_bundle(
        constants,
        source_version="offline validation fixture",
        source_uri="https://example.invalid/nist-codata-validation",
        retrieved_at="2026-04-30T00:00:00Z",
    )


def _build_opb_validation_bundle() -> Any:
    from sciona.physics_ingest.sources.opb import build_opb_wave0_bundle

    return build_opb_wave0_bundle(
        (
            {
                "problem_id": "opb:validation:newton-2",
                "title": "Newton second law",
                "latex": "F = m a",
                "data": {"fixture_rows": [{"m": 2, "a": 3, "F": 6}]},
            },
        ),
        source_version="offline validation fixture",
        source_uri="https://example.invalid/opb-validation",
        retrieved_at="2026-04-30T00:00:00Z",
    )


def _build_phy_srbench_validation_bundle() -> Any:
    from sciona.physics_ingest.sources.phy_srbench import (
        build_phy_srbench_wave0_bundle,
    )

    return build_phy_srbench_wave0_bundle(
        (
            {
                "task_id": "phy-srbench:validation:kepler",
                "name": "Kepler third law synthetic task",
                "sympy": "Eq(T**2, 4*pi**2*a**3/(G*M))",
                "dataset": {"train_rows": [{"a": 1.0, "T": 1.0}]},
                "evaluation_spec": {"metric": "r2", "threshold": 0.99},
            },
        ),
        source_version="offline validation fixture",
        source_uri="https://example.invalid/phy-srbench-validation",
        retrieved_at="2026-04-30T00:00:00Z",
        license_expression="offline validation fixture",
    )


def _build_theoria_validation_bundle() -> Any:
    from sciona.physics_ingest.sources.theoria import build_theoria_wave0_bundle

    return build_theoria_wave0_bundle(
        (
            {
                "problem_id": "theoria:validation:oscillator",
                "title": "Harmonic oscillator Euler-Lagrange equation",
                "latex": "F = -k x",
                "theory": "classical mechanics",
                "evaluation": {"metric": "symbolic_equivalence", "tolerance": 0},
            },
        ),
        source_version="offline validation fixture",
        source_uri="https://example.invalid/theoria-validation",
        retrieved_at="2026-04-30T00:00:00Z",
        license_expression="offline validation fixture",
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


def _text_value(row: Mapping[str, Any], *keys: str) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None:
            text = str(value)
            if text:
                return text
    return ""


def _publication_candidate_keys(row: Mapping[str, Any]) -> tuple[str, ...]:
    keys: list[str] = []
    for key_name in ("local_artifact_key", "artifact_key", "atom_name", "registry_name"):
        value = _text_value(row, key_name)
        if value and value not in keys:
            keys.append(value)
    return tuple(keys)


def _symbol_reference_values(row: Mapping[str, Any]) -> tuple[str, ...]:
    symbols: list[str] = []
    for key_name in (
        "variable_name",
        "symbol_name",
        "symbol",
        "source_symbol",
        "variable_id",
        "source_variable_id",
    ):
        value = _text_value(row, key_name)
        if value and value not in symbols:
            symbols.append(value)
    return tuple(symbols)


def _real_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


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
    "discover_changed_pdg_payload_fixture_paths",
    "discover_changed_symbolic_fixture_paths",
    "discover_git_changed_paths",
    "discover_pdg_payload_fixture_paths",
    "discover_symbolic_fixture_paths",
    "validate_pdg_payload",
    "validate_pdg_payload_file",
    "validate_source_adapter_coverage",
    "validate_source_adapter_data_artifact_seed_quality",
    "validate_source_execution_readiness",
    "validate_symbolic_publication_fixture",
]

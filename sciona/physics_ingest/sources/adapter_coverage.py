"""Side-effect-free source adapter coverage reporting."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from typing import Any, Mapping

from sciona.physics_ingest.sources._manifest import jsonable
from sciona.physics_ingest.sources.retrieval_plan import (
    SourceRetrievalManifest,
    build_physics_source_retrieval_manifest,
)


JSONDict = dict[str, Any]

REPORT_VERSION = "physics-source-adapter-coverage.v1"

_KNOWN_TARGET_INPUTS = {
    "curated_seed_records",
    "json_records",
    "raw_document",
    "raw_records",
}

_JOB_CONTRACTS: dict[str, tuple[str, ...]] = {
    "foundational_manual_seed.backfill": ("build_foundational_physics_backfill_bundle",),
    "hitran_lines.backfill": ("build_hitran_wave0_bundle",),
    "materials_project_documents.backfill": ("build_materials_project_wave0_bundle",),
    "nist_codata_constants.backfill": ("build_codata_wave0_bundle",),
    "nist_dlmf_equations.backfill": ("build_dlmf_wave0_bundle",),
    "opb_problem_payloads.backfill": ("build_opb_wave0_bundle",),
    "pdg_derivation_graph.backfill": ("parse_pdg_document",),
    "phy_srbench_payloads.backfill": ("build_phy_srbench_wave0_bundle",),
    "qudt_units_quantity_kinds.backfill": ("build_qudt_snapshot_manifest",),
    "theoria_payloads.backfill": ("build_theoria_wave0_bundle",),
    "wikidata_equation_candidates.backfill": (
        "build_snapshot_record",
        "build_wave0_candidate_records",
    ),
}


@dataclass(frozen=True)
class SourceAdapterCoverageDiagnostic:
    """Deterministic adapter coverage diagnostic."""

    severity: str
    code: str
    message: str
    job_id: str = ""
    adapter_module: str = ""

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceAdapterCoverageJob:
    """Adapter coverage for one retrieval manifest job."""

    job_id: str
    status: str
    adapter_module: str
    adapter_version: str
    source_family: str
    target_adapter_input: str
    supported: bool
    offline: bool
    manual: bool
    adapter_metadata: Mapping[str, Any]
    offline_contract: Mapping[str, Any]
    diagnostics: tuple[SourceAdapterCoverageDiagnostic, ...] = ()

    def to_dict(self) -> JSONDict:
        return jsonable(self)


@dataclass(frozen=True)
class SourceAdapterCoverageReport:
    """JSON-safe adapter coverage report for the retrieval manifest."""

    report_version: str
    manifest_version: str
    snapshot_key_prefix: str
    summary: Mapping[str, Any]
    jobs: tuple[SourceAdapterCoverageJob, ...]
    diagnostics: tuple[SourceAdapterCoverageDiagnostic, ...] = ()

    def to_dict(self) -> JSONDict:
        return jsonable(self)


def build_source_adapter_coverage_report(
    manifest: SourceRetrievalManifest | Mapping[str, Any] | None = None,
) -> SourceAdapterCoverageReport:
    """Report retrieval job adapter metadata without network or database IO."""

    manifest_dict = _manifest_to_dict(
        manifest or build_physics_source_retrieval_manifest()
    )
    jobs = [
        _coverage_job(job)
        for job in sorted(
            _manifest_jobs(manifest_dict),
            key=lambda item: (
                _int_value(item.get("priority"), 100),
                str(item.get("job_id", "")),
            ),
        )
    ]
    diagnostics = tuple(
        diagnostic for job in jobs for diagnostic in job.diagnostics
    )
    summary = {
        "total_jobs": len(jobs),
        "covered": _count_status(jobs, "covered"),
        "metadata_only": _count_status(jobs, "metadata_only"),
        "blocked": _count_status(jobs, "blocked"),
        "manual": sum(1 for job in jobs if job.manual),
        "offline": sum(1 for job in jobs if job.offline),
        "diagnostic_count": len(diagnostics),
        "by_source_family": _count_by_field(jobs, "source_family"),
        "by_target_adapter_input": _count_by_field(jobs, "target_adapter_input"),
        "by_status": {
            status: _count_status(jobs, status)
            for status in ("blocked", "covered", "metadata_only")
        },
        "by_supported": _count_by_bool(jobs, "supported"),
        "by_offline": _count_by_bool(jobs, "offline"),
        "by_manual": _count_by_bool(jobs, "manual"),
        "by_adapter_module": _count_by_field(jobs, "adapter_module"),
    }
    return SourceAdapterCoverageReport(
        report_version=REPORT_VERSION,
        manifest_version=str(manifest_dict.get("manifest_version", "")),
        snapshot_key_prefix=str(manifest_dict.get("snapshot_key_prefix", "")),
        summary=summary,
        jobs=tuple(jobs),
        diagnostics=diagnostics,
    )


def build_source_adapter_coverage_report_dict(
    manifest: SourceRetrievalManifest | Mapping[str, Any] | None = None,
) -> JSONDict:
    """Return ``build_source_adapter_coverage_report(manifest).to_dict()``."""

    return build_source_adapter_coverage_report(manifest).to_dict()


def _coverage_job(job: Mapping[str, Any]) -> SourceAdapterCoverageJob:
    job_id = str(job.get("job_id", ""))
    adapter_module = str(job.get("adapter_name") or job.get("adapter_module") or "")
    expected_version = str(job.get("adapter_version", ""))
    target_adapter_input = str(job.get("target_adapter_input", ""))
    manual = target_adapter_input == "curated_seed_records" or str(
        job.get("source_system", "")
    ) == "manual"
    diagnostics: list[SourceAdapterCoverageDiagnostic] = []
    module: Any | None = None
    import_error = ""

    if not adapter_module:
        diagnostics.append(
            _diagnostic(
                "error",
                "missing_adapter_module",
                "retrieval job does not declare an adapter module",
                job_id,
                adapter_module,
            )
        )
    else:
        try:
            module = import_module(adapter_module)
        except Exception as exc:  # pragma: no cover - message shape varies by Python
            import_error = f"{type(exc).__name__}: {exc}"
            diagnostics.append(
                _diagnostic(
                    "error",
                    "missing_adapter_module",
                    "adapter module could not be imported",
                    job_id,
                    adapter_module,
                )
            )

    discovered_version = _adapter_version(module) if module is not None else ""
    if not expected_version:
        diagnostics.append(
            _diagnostic(
                "error",
                "missing_adapter_version",
                "retrieval job does not declare an adapter version",
                job_id,
                adapter_module,
            )
        )
    elif module is not None and not discovered_version:
        diagnostics.append(
            _diagnostic(
                "error",
                "missing_adapter_version",
                "adapter module does not expose adapter version metadata",
                job_id,
                adapter_module,
            )
        )
    elif discovered_version and discovered_version != expected_version:
        diagnostics.append(
            _diagnostic(
                "error",
                "adapter_version_mismatch",
                "retrieval job adapter version does not match imported metadata",
                job_id,
                adapter_module,
            )
        )

    if target_adapter_input not in _KNOWN_TARGET_INPUTS:
        diagnostics.append(
            _diagnostic(
                "error",
                "unknown_target_adapter_input",
                "retrieval job target adapter input is not recognized",
                job_id,
                adapter_module,
            )
        )

    contract_names = _JOB_CONTRACTS.get(job_id, ())
    available_contracts = tuple(
        name
        for name in contract_names
        if module is not None and callable(getattr(module, name, None))
    )
    if not available_contracts:
        diagnostics.append(
            _diagnostic(
                "warning" if module is not None else "error",
                "missing_builder_readiness_contract",
                "adapter module does not expose the known offline builder/readiness contract",
                job_id,
                adapter_module,
            )
        )

    has_error = any(diagnostic.severity == "error" for diagnostic in diagnostics)
    offline = bool(available_contracts)
    status = "blocked" if has_error else "covered" if offline else "metadata_only"
    return SourceAdapterCoverageJob(
        job_id=job_id,
        status=status,
        adapter_module=adapter_module,
        adapter_version=expected_version,
        source_family=str(job.get("source_family", "")),
        target_adapter_input=target_adapter_input,
        supported=not has_error,
        offline=offline,
        manual=manual,
        adapter_metadata={
            "importable": module is not None,
            "adapter_name": _adapter_name(module) if module is not None else "",
            "adapter_version": discovered_version,
            "import_error": import_error,
        },
        offline_contract={
            "known_contract": list(contract_names),
            "available_contract": list(available_contracts),
            "contract_present": bool(available_contracts),
        },
        diagnostics=tuple(diagnostics),
    )


def _manifest_to_dict(
    manifest: SourceRetrievalManifest | Mapping[str, Any],
) -> JSONDict:
    if isinstance(manifest, Mapping):
        return jsonable(manifest)
    return manifest.to_dict()


def _manifest_jobs(manifest: Mapping[str, Any]) -> tuple[Mapping[str, Any], ...]:
    jobs = manifest.get("jobs", ())
    if not isinstance(jobs, list | tuple):
        return ()
    return tuple(job for job in jobs if isinstance(job, Mapping))


def _adapter_name(module: Any) -> str:
    return str(
        getattr(module, "ADAPTER_NAME", "")
        or getattr(module, "PDG_ADAPTER_NAME", "")
    )


def _adapter_version(module: Any) -> str:
    return str(
        getattr(module, "ADAPTER_VERSION", "")
        or getattr(module, "PDG_ADAPTER_VERSION", "")
    )


def _diagnostic(
    severity: str,
    code: str,
    message: str,
    job_id: str,
    adapter_module: str,
) -> SourceAdapterCoverageDiagnostic:
    return SourceAdapterCoverageDiagnostic(
        severity=severity,
        code=code,
        message=message,
        job_id=job_id,
        adapter_module=adapter_module,
    )


def _count_status(jobs: list[SourceAdapterCoverageJob], status: str) -> int:
    return sum(1 for job in jobs if job.status == status)


def _count_by_field(
    jobs: list[SourceAdapterCoverageJob],
    field_name: str,
) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        value = str(getattr(job, field_name))
        counts[value] = counts.get(value, 0) + 1
    return {key: counts[key] for key in sorted(counts)}


def _count_by_bool(
    jobs: list[SourceAdapterCoverageJob],
    field_name: str,
) -> dict[str, int]:
    return {
        "false": sum(1 for job in jobs if not bool(getattr(job, field_name))),
        "true": sum(1 for job in jobs if bool(getattr(job, field_name))),
    }


def _int_value(value: Any, default: int) -> int:
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

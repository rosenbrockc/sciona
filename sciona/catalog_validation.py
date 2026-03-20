"""Release-style validation for source-derived architect catalog coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sciona.architect.catalog import CatalogReport, PrimitiveCatalog, seed_builtin_primitives
from sciona.architect.source_catalog import (
    audit_source_registration_alignment,
    seed_catalog_from_sources,
)
from sciona.config import AgeomConfig
from sciona.sources import load_sources, resolve_source


def _catalog_validation_config() -> AgeomConfig:
    """Use repo defaults instead of operator-local dotenv overrides."""
    return AgeomConfig(_env_file=None)


def _format_catalog_coverage_summary(summary: dict[str, Any]) -> str:
    return (
        f"resolved={int(summary.get('resolved_sources', 0) or 0)}/"
        f"{int(summary.get('configured_sources', 0) or 0)} "
        f"added={int(summary.get('source_added', 0) or 0)}/"
        f"{int(summary.get('source_candidates', 0) or 0)} "
        f"missing={len(summary.get('missing_sources', []) or [])} "
        f"zero={len(summary.get('zero_candidate_sources', []) or [])}"
    )


def _format_catalog_alignment_summary(alignment: dict[str, Any]) -> str:
    return (
        f"severity={str(alignment.get('highest_severity', '') or 'healthy')} "
        f"matched={int(alignment.get('matched_total', 0) or 0)} "
        f"registry_only={int(alignment.get('registry_only_total', 0) or 0)} "
        f"ast_only={int(alignment.get('ast_only_total', 0) or 0)} "
        f"drift={len(alignment.get('drift_sources', []) or [])}"
    )


def _format_catalog_warning_summary(summary: dict[str, Any]) -> str:
    warnings = summary.get("warnings", []) if isinstance(summary.get("warnings", []), list) else []
    high_sources = summary.get("high_severity_sources", []) if isinstance(summary.get("high_severity_sources", []), list) else []
    medium_sources = summary.get("medium_severity_sources", []) if isinstance(summary.get("medium_severity_sources", []), list) else []
    return (
        f"warnings={len(warnings)} "
        f"high={len(high_sources)} "
        f"medium={len(medium_sources)}"
    )


def _source_rows(config: Any, *, base_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for source in config.sources:
        resolved = ""
        exists = False
        error = ""
        try:
            path = resolve_source(source, base_dir)
            resolved = str(path)
            exists = path.exists()
            if not exists:
                missing.append(source.name)
                error = "resolved path missing"
        except Exception as exc:
            missing.append(source.name)
            error = str(exc)
        rows.append(
            {
                "source": source.name,
                "package": source.package,
                "resolved_path": resolved,
                "exists": exists,
                "error": error,
            }
        )
    return rows, missing


async def run_catalog_validation(output_dir: str | Path) -> dict[str, Any]:
    """Run deterministic validation over configured source-derived catalog coverage."""
    config = _catalog_validation_config()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sources_cfg = load_sources(config.sources_file)
    source_rows, missing_sources = _source_rows(sources_cfg, base_dir=Path.cwd())

    catalog = PrimitiveCatalog()
    seed_builtin_primitives(catalog)
    report = CatalogReport()
    derived_added = 0
    if sources_cfg.sources:
        derived_added = seed_catalog_from_sources(
            catalog,
            config=sources_cfg,
            base_dir=Path.cwd(),
            include_live_registries=False,
            report=report,
        )
    alignment = audit_source_registration_alignment(
        config=sources_cfg,
        base_dir=Path.cwd(),
    )

    zero_candidate_sources = sorted(
        row["source"]
        for row in source_rows
        if int(
            (
                report.source_breakdown.get(str(row["source"]), {}).get("ast_candidates", 0)
                or 0
            )
            + (
                report.source_breakdown.get(str(row["source"]), {}).get(
                    "live_registry_candidates", 0
                )
                or 0
            )
        )
        == 0
    )

    violations: list[str] = []
    if not sources_cfg.sources:
        violations.append("no_configured_sources")
    if missing_sources:
        violations.extend(f"missing_source:{name}" for name in missing_sources)
    if not any(bool(row["exists"]) for row in source_rows):
        violations.append("no_resolved_sources")
    if report.total_candidates <= 0:
        violations.append("no_source_candidates")
    if zero_candidate_sources:
        violations.extend(f"source_no_candidates:{name}" for name in zero_candidate_sources)
    highest_severity = str(alignment.get("highest_severity", "") or "").strip().lower()
    high_severity_sources = sorted(
        str(row.get("source", "") or "")
        for row in alignment.get("rows", [])
        if isinstance(row, dict) and str(row.get("severity", "") or "").strip().lower() == "high"
    )
    medium_severity_sources = sorted(
        str(row.get("source", "") or "")
        for row in alignment.get("rows", [])
        if isinstance(row, dict) and str(row.get("severity", "") or "").strip().lower() == "medium"
    )
    warnings: list[str] = []
    warnings.extend(f"high_alignment_drift:{name}" for name in high_severity_sources)
    warnings.extend(f"medium_alignment_drift:{name}" for name in medium_severity_sources)
    if highest_severity == "critical":
        violations.append("critical_alignment_drift")

    status = "passed" if not violations else "failed"
    summary = {
        "status": status,
        "configured_sources": len(sources_cfg.sources),
        "resolved_sources": sum(1 for row in source_rows if row["exists"]),
        "missing_sources": sorted(missing_sources),
        "zero_candidate_sources": zero_candidate_sources,
        "catalog_size": catalog.size,
        "source_candidates": report.total_candidates,
        "source_added": derived_added,
        "source_merged": report.merged,
        "source_structural_skips": report.structural_skips,
        "source_witness_doc_fallbacks": report.source_witness_doc_fallbacks,
        "source_witness_signature_fallbacks": report.source_witness_signature_fallbacks,
        "source_rows": source_rows,
        "source_breakdown": report.source_breakdown,
        "alignment": alignment,
        "high_severity_sources": high_severity_sources,
        "medium_severity_sources": medium_severity_sources,
        "warnings": warnings,
        "violations": violations,
    }
    summary["coverage_summary"] = _format_catalog_coverage_summary(summary)
    summary["alignment_summary"] = _format_catalog_alignment_summary(alignment)
    summary["warning_summary"] = _format_catalog_warning_summary(summary)
    report_path = out_dir / "catalog_validation.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary

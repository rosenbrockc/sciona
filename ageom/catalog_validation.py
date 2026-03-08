"""Release-style validation for source-derived architect catalog coverage."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.architect.catalog import CatalogReport, PrimitiveCatalog, seed_builtin_primitives
from ageom.architect.source_catalog import (
    audit_source_registration_alignment,
    seed_catalog_from_sources,
)
from ageom.config import AgeomConfig
from ageom.sources import load_sources, resolve_source


def _catalog_validation_config() -> AgeomConfig:
    """Use repo defaults instead of operator-local dotenv overrides."""
    return AgeomConfig(_env_file=None)


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
        "violations": violations,
    }
    report_path = out_dir / "catalog_validation.json"
    report_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["report"] = str(report_path)
    return summary

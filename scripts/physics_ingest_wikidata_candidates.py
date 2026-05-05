#!/usr/bin/env python
"""Build and optionally apply a Wikidata physics candidate ingestion plan."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_TABLE_MODES = {
    "physics_ingest_snapshots": "upsert",
    "physics_equation_candidates": "upsert",
}


def main() -> int:
    from sciona.physics_ingest.backfill import build_physics_ingest_backfill_report
    from sciona.physics_ingest.pipeline import run_physics_publication_pipeline
    from sciona.physics_ingest.sources.retrieval_plan import (
        build_physics_source_retrieval_run_plan,
    )
    from sciona.physics_ingest.sources.wikidata import build_snapshot_record

    parser = argparse.ArgumentParser(
        description=(
            "Build a write plan for a saved Wikidata physics candidate list. "
            "Use --apply only when Supabase credentials are present."
        )
    )
    parser.add_argument(
        "--candidate-dir",
        type=Path,
        default=None,
        help=(
            "Directory containing query.sparql, response.json, and "
            "wave0_candidate_records.json. Defaults to the latest "
            "output/physics_wikidata_physics_candidate_list_* directory."
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for ingestion artifacts.",
    )
    parser.add_argument(
        "--source-version",
        default="wikidata-physics-candidates-2026-05-05-limit-50",
        help="Source version string for the snapshot row.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the write plan through Supabase/PostgREST credentials.",
    )
    args = parser.parse_args()

    candidate_dir = args.candidate_dir or _latest_candidate_dir()
    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    query = _read_text(candidate_dir / "query.sparql")
    response = _read_json(candidate_dir / "response.json")
    candidate_rows = _read_json(candidate_dir / "wave0_candidate_records.json")
    if not isinstance(candidate_rows, list):
        raise ValueError("wave0_candidate_records.json must contain a list")

    snapshot_row = build_snapshot_record(
        query=query,
        response=response,
        source_version=args.source_version,
        provenance_summary=(
            "Wikidata physics-scoped ingestion queue built from items classified "
            "under Physics equations or physical law roots."
        ),
    )
    source_bundle = {
        "bundle_key": f"wikidata-physics-candidates:{args.source_version}",
        "snapshot_row": snapshot_row,
        "candidate_rows": candidate_rows,
    }
    source_run_plan = build_physics_source_retrieval_run_plan(
        job_id="wikidata_equation_candidates.backfill",
        dry_run=False,
        limit=_candidate_query_limit(candidate_dir),
    )

    pipeline_result = run_physics_publication_pipeline(
        source_bundles=[source_bundle],
        table_modes=DEFAULT_TABLE_MODES,
        dry_run=False,
    )
    backfill_report = build_physics_ingest_backfill_report(
        source_bundles=[source_bundle],
        source_retrieval_run_plan=source_run_plan,
        table_modes=DEFAULT_TABLE_MODES,
        include_source_request_envelopes=True,
        include_source_runtime_execution_preflight=True,
        include_publication_write_preflight=True,
        include_audit_artifact_manifests=True,
        include_rows=True,
    )
    apply_report = _apply_if_requested(
        apply_requested=args.apply,
        write_plan=pipeline_result.write_plan,
    )

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "candidate_dir": str(candidate_dir),
        "out_dir": str(out_dir),
        "source_version": args.source_version,
        "candidate_count": len(candidate_rows),
        "snapshot_id": pipeline_result.planned_source_bundles[0]["snapshot_row"][
            "snapshot_id"
        ],
        "table_row_counts": dict(
            pipeline_result.write_plan.audit_summary.planned_row_counts
        ),
        "write_plan_total_row_count": (
            pipeline_result.write_plan.audit_summary.total_row_count
        ),
        "write_plan_table_modes": dict(DEFAULT_TABLE_MODES),
        "apply": apply_report["summary"],
        "artifact_paths": {
            "summary": str(out_dir / "summary.json"),
            "source_bundle": str(out_dir / "source_bundle.json"),
            "planned_source_bundle": str(out_dir / "planned_source_bundle.json"),
            "write_plan": str(out_dir / "write_plan.json"),
            "insert_rows_by_table": str(out_dir / "insert_rows_by_table.json"),
            "backfill_report": str(out_dir / "backfill_report.json"),
            "apply_report": str(out_dir / "apply_report.json"),
            "learnings": str(out_dir / "ingestion_template_learnings.md"),
        },
    }

    _write_json(out_dir / "source_bundle.json", source_bundle)
    _write_json(
        out_dir / "planned_source_bundle.json",
        pipeline_result.planned_source_bundles[0],
    )
    _write_json(out_dir / "write_plan.json", pipeline_result.write_plan.to_dict())
    _write_json(
        out_dir / "insert_rows_by_table.json",
        pipeline_result.write_plan.to_insert_rows(),
    )
    _write_json(out_dir / "pipeline_result.json", pipeline_result.to_dict())
    _write_json(out_dir / "backfill_report.json", backfill_report)
    _write_json(out_dir / "apply_report.json", apply_report)
    _write_text(out_dir / "ingestion_template_learnings.md", _learnings(summary))
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if apply_report["summary"]["ok"] else 2


def _apply_if_requested(*, apply_requested: bool, write_plan: Any) -> dict[str, Any]:
    if not apply_requested:
        return {
            "summary": {
                "requested": False,
                "performed": False,
                "ok": True,
                "reason": "apply_not_requested",
            }
        }

    credentials = _supabase_credentials()
    if credentials is None:
        return {
            "summary": {
                "requested": True,
                "performed": False,
                "ok": False,
                "reason": "missing_supabase_credentials",
                "required_env": [
                    "SUPABASE_URL or SCIONA_SUPABASE_URL",
                    (
                        "SUPABASE_SERVICE_ROLE_KEY, SUPABASE_SERVICE_KEY, "
                        "SCIONA_SUPABASE_SERVICE_ROLE_KEY, or "
                        "SCIONA_SUPABASE_SERVICE_KEY"
                    ),
                ],
            }
        }

    from supabase import create_client

    from sciona.physics_ingest.supabase_adapter import apply_publication_supabase_write

    client = create_client(credentials["url"], credentials["key"])
    result = apply_publication_supabase_write(
        client,
        write_plan,
        dry_run=False,
    )
    return {
        "summary": {
            "requested": True,
            "performed": True,
            "ok": not result.has_errors,
            "reason": "applied" if not result.has_errors else "apply_errors",
        },
        "write_result": result.to_dict(),
    }


def _supabase_credentials() -> dict[str, str] | None:
    url = os.environ.get("SUPABASE_URL") or os.environ.get("SCIONA_SUPABASE_URL")
    key = (
        os.environ.get("SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SUPABASE_SERVICE_KEY")
        or os.environ.get("SCIONA_SUPABASE_SERVICE_ROLE_KEY")
        or os.environ.get("SCIONA_SUPABASE_SERVICE_KEY")
    )
    if not url or not key:
        return None
    return {"url": url, "key": key}


def _latest_candidate_dir() -> Path:
    candidates = sorted(Path("output").glob("physics_wikidata_physics_candidate_list_*"))
    if not candidates:
        raise FileNotFoundError(
            "no output/physics_wikidata_physics_candidate_list_* directories found"
        )
    return candidates[-1]


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path("output") / f"physics_wikidata_physics_ingestion_{stamp}"


def _candidate_query_limit(candidate_dir: Path) -> int:
    summary_path = candidate_dir / "summary.json"
    if not summary_path.exists():
        return 500
    summary = _read_json(summary_path)
    if not isinstance(summary, Mapping):
        return 500
    value = summary.get("query_limit")
    try:
        return max(int(value), 1)
    except (TypeError, ValueError):
        return 500


def _read_json(path: Path) -> Any:
    return json.loads(_read_text(path))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.write_text(value, encoding="utf-8")


def _learnings(summary: Mapping[str, Any]) -> str:
    table_counts = summary.get("table_row_counts", {})
    return f"""# Wikidata Physics Ingestion Learnings

Generated at: {summary.get("generated_at")}

## Run Summary

- Candidate source: `{summary.get("candidate_dir")}`
- Candidate count: {summary.get("candidate_count")}
- Planned storage rows: {summary.get("write_plan_total_row_count")}
- Table counts: `{json.dumps(table_counts, sort_keys=True)}`
- Table modes: `{json.dumps(summary.get("write_plan_table_modes", {}), sort_keys=True)}`
- Apply status: `{json.dumps(summary.get("apply", {}), sort_keys=True)}`

## Template Notes

1. Start broad discovery with source-native evidence, but create the first
   ingestion queue from a scoped query. For Wikidata, `P2534` alone is too broad;
   it pulls mathematical objects and number curiosities. The first queue should
   use class roots for Physics equations and physical law.
2. Preserve raw source formula payloads exactly. For Wikidata this means storing
   the MathML in `raw_formula` and using `formula_plain_text` only as a parse
   hint in `source_payload`.
3. Generate deterministic snapshot and candidate IDs before any apply attempt.
   The same candidate list should produce the same `snapshot_id` and
   `candidate_id` values.
4. Use upsert modes for source snapshots and candidates. Source ingestion should
   be repeatable while parser and review metadata improve.
5. Treat write-plan creation, storage preflight, and storage apply as separate
   phases. If Supabase credentials or local stack are absent, the write plan is
   still a durable handoff artifact.
6. Record the exact query, response, source bundle, planned rows, and apply
   result in the output directory. This makes the source backfill replayable and
   auditable.

## Next Refinements

- Add a standard CLI entrypoint for source-specific ingestion queues instead of
  one-off scripts.
- Add post-apply verification queries for `physics_ingest_snapshots` and
  `physics_equation_candidates` by deterministic IDs.
- Add a normalization pass over `formula_plain_text` so simple laws can move
  from `raw_imported` to parsed symbolic drafts after source capture.
"""


if __name__ == "__main__":
    raise SystemExit(main())

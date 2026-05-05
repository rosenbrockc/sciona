#!/usr/bin/env python
"""Build and optionally apply a first PDG ingestion wave through CDG rows."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys
from typing import Any, Iterable, Mapping
from uuid import NAMESPACE_URL, uuid5

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


DEFAULT_PDG_PAYLOAD = (
    REPO_ROOT
    / "tests"
    / "physics_ingest"
    / "fixtures"
    / "pdg_payloads"
    / "solve_substitute_chain.pdg.json"
)
DEFAULT_TABLE_MODES = {
    "physics_ingest_snapshots": "upsert",
    "physics_equation_candidates": "upsert",
    "artifacts": "upsert",
    "artifact_versions": "upsert",
    "artifact_symbolic_expressions": "upsert",
    "artifact_relationships": "upsert",
    "artifact_cdg_nodes": "upsert",
    "artifact_cdg_edges": "upsert",
    "artifact_cdg_bindings": "upsert",
}

_ARTIFACT_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.pdg_first_wave.artifact")
_VERSION_NAMESPACE = uuid5(NAMESPACE_URL, "sciona.physics_ingest.pdg_first_wave.version")
_EXPRESSION_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.pdg_first_wave.expression",
)
_RELATIONSHIP_NAMESPACE = uuid5(
    NAMESPACE_URL,
    "sciona.physics_ingest.pdg_first_wave.relationship",
)


def main() -> int:
    from sciona.physics_ingest.ids import plan_source_bundle_ids
    from sciona.physics_ingest.pdg_cdg import (
        PDGCDGArtifactEnvelope,
        build_pdg_publication_write_rows,
        build_pdg_relationship_ingest,
        validate_pdg_cdg_publication_graph,
    )
    from sciona.physics_ingest.pipeline import run_physics_publication_pipeline
    from sciona.physics_ingest.sources.pdg import parse_pdg_document
    from sciona.physics_ingest.write_plan import (
        build_publication_write_plan,
        merge_publication_insert_rows,
    )

    parser = argparse.ArgumentParser(
        description="Materialize a first PDG payload through symbolic rows and CDGs."
    )
    parser.add_argument(
        "--payload",
        type=Path,
        default=DEFAULT_PDG_PAYLOAD,
        help="PDG JSON payload to ingest.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory for generated write artifacts.",
    )
    parser.add_argument(
        "--source-version",
        default="pdg-first-wave-2026-05-05",
        help="Source version string for the PDG snapshot.",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply the write plan through Supabase/PostgREST credentials.",
    )
    args = parser.parse_args()

    payload_path = args.payload
    payload = _read_json(payload_path)
    if not isinstance(payload, Mapping):
        raise ValueError("PDG payload root must be a JSON object")

    bundle = parse_pdg_document(
        payload,
        source_uri=str(payload_path),
        source_version=args.source_version,
        license_expression=str(payload.get("license_expression") or ""),
        provenance_summary=(
            "First local PDG ingestion wave from a checked-in derivation fixture."
        ),
    )
    if not bundle.equations:
        raise ValueError("PDG payload has no equation nodes")
    if not bundle.inference_edges:
        raise ValueError("PDG payload has no inference edges")

    source_bundle = {
        "bundle_key": f"pdg-first-wave:{args.source_version}:{payload_path.name}",
        "snapshot_row": bundle.snapshot_row,
        "candidate_rows": bundle.candidate_rows(),
    }
    _, planned_source_bundles = plan_source_bundle_ids([source_bundle])
    planned_source_bundle = planned_source_bundles[0]
    candidate_id_by_node_id = {
        str(row["source_candidate_id"]): str(row["candidate_id"])
        for row in planned_source_bundle["candidate_rows"]
    }

    symbolic_rows = _equation_symbolic_rows(
        bundle.equations,
        candidate_id_by_node_id=candidate_id_by_node_id,
        source_version=args.source_version,
    )
    expression_bindings = {
        node_id: {
            "expression_id": rows["expression"]["expression_id"],
            "artifact_id": rows["artifact"]["artifact_id"],
            "version_id": rows["version"]["version_id"],
            "label": rows["label"],
            "metadata": {
                "bound_artifact_fqdn": rows["artifact"]["fqdn"],
                "bound_version_content_hash": rows["version"]["content_hash"],
                "binding_confidence": 0.75,
                "binding_source": "pdg_first_wave",
            },
        }
        for node_id, rows in symbolic_rows.items()
    }
    relationship_result = build_pdg_relationship_ingest(
        bundle,
        expression_bindings_by_pdg_node_id=expression_bindings,
    )
    pdg_publication_rows = build_pdg_publication_write_rows(
        relationship_result,
        cdg_artifact_envelope=PDGCDGArtifactEnvelope(
            fqdn_prefix="physics.pdg.first_wave.cdg",
            semver="2026.5.5",
            namespace_root="physics",
            namespace_path="pdg/first_wave/cdg",
            source_package="pdg",
            source_module_path="pdg.derivations",
            source_symbol_prefix="pdg_first_wave_cdg",
            status="draft",
            visibility_tier="internal",
            description="First-wave PDG-derived CDG from checked-in source payload",
            is_latest=True,
        ),
    )
    pdg_rows = _with_stable_relationship_ids(pdg_publication_rows.to_insert_rows())
    graph_diagnostics = validate_pdg_cdg_publication_graph(pdg_rows)
    if graph_diagnostics:
        errors = [
            row for row in graph_diagnostics if row.get("severity", "error") == "error"
        ]
        if errors:
            raise ValueError(f"PDG CDG graph validation failed: {errors}")

    source_pipeline = run_physics_publication_pipeline(
        source_bundles=[source_bundle],
        table_modes={
            "physics_ingest_snapshots": "upsert",
            "physics_equation_candidates": "upsert",
        },
        dry_run=False,
    )
    insert_rows = merge_publication_insert_rows(
        source_pipeline.write_plan.to_insert_rows(),
        _symbolic_rows_by_table(symbolic_rows.values()),
        pdg_rows,
    )
    write_plan = build_publication_write_plan(
        insert_rows,
        table_modes=DEFAULT_TABLE_MODES,
    )
    apply_report = _apply_if_requested(args.apply, write_plan)

    out_dir = args.out_dir or _default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "payload": str(payload_path),
        "out_dir": str(out_dir),
        "source_version": args.source_version,
        "snapshot_id": planned_source_bundle["snapshot_row"]["snapshot_id"],
        "equation_count": len(bundle.equations),
        "inference_edge_count": len(bundle.inference_edges),
        "relationship_summary": dict(relationship_result.summary),
        "pdg_publication_summary": dict(pdg_publication_rows.summary),
        "graph_diagnostic_count": len(graph_diagnostics),
        "table_row_counts": dict(write_plan.audit_summary.planned_row_counts),
        "write_plan_total_row_count": write_plan.audit_summary.total_row_count,
        "write_plan_table_modes": dict(DEFAULT_TABLE_MODES),
        "apply": apply_report["summary"],
    }

    _write_json(out_dir / "source_bundle.json", source_bundle)
    _write_json(out_dir / "planned_source_bundle.json", planned_source_bundle)
    _write_json(out_dir / "symbolic_rows_by_pdg_node.json", symbolic_rows)
    _write_json(out_dir / "pdg_publication_rows.json", pdg_rows)
    _write_json(out_dir / "graph_diagnostics.json", list(graph_diagnostics))
    _write_json(out_dir / "insert_rows_by_table.json", write_plan.to_insert_rows())
    _write_json(out_dir / "write_plan.json", write_plan.to_dict())
    _write_json(out_dir / "apply_report.json", apply_report)
    _write_json(out_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if apply_report["summary"]["ok"] else 2


def _equation_symbolic_rows(
    equations: Iterable[Any],
    *,
    candidate_id_by_node_id: Mapping[str, str],
    source_version: str,
) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for equation in equations:
        node_id = str(equation.node_id)
        label = str(equation.label or node_id)
        formula = str(equation.formula or "")
        artifact_id = str(uuid5(_ARTIFACT_NAMESPACE, node_id))
        version_id = str(uuid5(_VERSION_NAMESPACE, node_id))
        expression_id = str(uuid5(_EXPRESSION_NAMESPACE, node_id))
        fqdn = f"physics.pdg.first_wave.equation.{_fqdn_token(node_id)}"
        content_hash = _sha256_json(
            {
                "source": "physics_derivation_graph",
                "source_version": source_version,
                "node_id": node_id,
                "formula": formula,
                "label": label,
            }
        )
        rows[node_id] = {
            "label": label,
            "artifact": {
                "artifact_id": artifact_id,
                "artifact_kind": "atom",
                "fqdn": fqdn,
                "namespace_root": "physics",
                "namespace_path": "pdg/first_wave/equations",
                "source_package": "pdg",
                "source_module_path": "pdg.equations",
                "source_symbol": _fqdn_token(node_id),
                "status": "draft",
                "visibility_tier": "internal",
                "description": f"PDG equation node: {label}",
                "source_kind": "generated",
                "is_stochastic": False,
                "is_ffi": False,
                "is_publishable": False,
                "topo_hash": "",
                "top_level_input_arity": 0,
                "top_level_output_arity": 1,
                "leaf_count": 1,
                "verified_leaf_coverage": 0.0,
            },
            "version": {
                "version_id": version_id,
                "artifact_id": artifact_id,
                "content_hash": content_hash,
                "semver": "2026.5.5",
                "is_latest": True,
                "s3_key": "",
                "fingerprint": content_hash,
            },
            "expression": {
                "expression_id": expression_id,
                "artifact_id": artifact_id,
                "version_id": version_id,
                "candidate_id": candidate_id_by_node_id.get(node_id),
                "expression_kind": "equation",
                "expression_role": "primary",
                "raw_formula": formula,
                "raw_formula_format": str(equation.formula_format or "plain_text"),
                "source_expression_id": node_id,
                "parse_status": "raw_imported",
                "parse_confidence": 0.5 if formula else 0.0,
                "review_status": "needs_human",
                "validation_status": "unknown",
                "mechanism_tags": list(equation.mechanism_tags),
                "behavioral_archetypes": list(equation.behavioral_archetypes),
                "assumptions_json": {"assumptions": list(equation.assumptions)},
                "evidence_json": {
                    "source_system": "physics_derivation_graph",
                    "pdg_node_id": node_id,
                    "source_version": source_version,
                    "label": label,
                },
            },
        }
    return rows


def _symbolic_rows_by_table(rows: Iterable[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    artifacts: list[dict[str, Any]] = []
    versions: list[dict[str, Any]] = []
    expressions: list[dict[str, Any]] = []
    for row in rows:
        artifacts.append(dict(row["artifact"]))
        versions.append(dict(row["version"]))
        expressions.append(
            {
                key: value
                for key, value in dict(row["expression"]).items()
                if value is not None
            }
        )
    return {
        "artifacts": artifacts,
        "artifact_versions": versions,
        "artifact_symbolic_expressions": expressions,
    }


def _with_stable_relationship_ids(
    rows_by_table: Mapping[str, Iterable[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    copied: dict[str, list[dict[str, Any]]] = {}
    for table, rows in rows_by_table.items():
        copied[table] = [dict(row) for row in rows]
    for row in copied.get("artifact_relationships", []):
        if row.get("relationship_id"):
            continue
        seed = {
            "source_kind": row.get("source_kind"),
            "relationship_kind": row.get("relationship_kind"),
            "source_expression_id": row.get("source_expression_id"),
            "target_expression_id": row.get("target_expression_id"),
            "source_node_id": row.get("source_node_id"),
            "target_node_id": row.get("target_node_id"),
            "inference_rule_id": row.get("inference_rule_id"),
        }
        row["relationship_id"] = str(uuid5(_RELATIONSHIP_NAMESPACE, _canonical_json(seed)))
    return copied


def _apply_if_requested(apply_requested: bool, write_plan: Any) -> dict[str, Any]:
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


def _default_out_dir() -> Path:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return REPO_ROOT / "output" / f"physics_pdg_first_wave_{stamp}"


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def _canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_json(data: Any) -> str:
    return hashlib.sha256(_canonical_json(data).encode("utf-8")).hexdigest()


def _fqdn_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "_", value).strip("_").lower()
    return token or "equation"


if __name__ == "__main__":
    raise SystemExit(main())

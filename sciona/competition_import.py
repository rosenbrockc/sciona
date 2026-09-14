"""Transactional draft intake for competition CDGs, without granting approval.

The version-bound asset-integrity record retains the complete input documents.
Relational bindings are a projection of active, named targets only: the legacy
SQL enums cannot represent unresolved or approximate binding states.
"""

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from psycopg import sql
from psycopg.types.json import Jsonb

from sciona.competition_promotion import CompetitionCandidate


FORMAT = "competition-intake.v1"
TABLE_KEYS = {
    "artifact_versions": ("version_id",),
    "artifact_cdg_nodes": ("version_id", "node_id"),
    "artifact_cdg_edges": ("version_id", "source_id", "target_id", "output_name", "input_name"),
    "artifact_cdg_bindings": ("binding_id",),
    "artifact_io_specs": ("io_spec_id",),
    "artifact_audit_evidence": ("evidence_id",),
}
JSON_FIELDS = {"details", "alternatives", "evidence_summary", "risk_dimensions"}
ACTIONS = {"precondition", "replace_stage", "split_stage", "insert_correction", "gate_or_validate", "smooth_or_aggregate", "branch_and_compare"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _id(value: str) -> str:
    return str(uuid5(NAMESPACE_URL, FORMAT + "/" + value))


@dataclass(frozen=True)
class CompetitionImportPlan:
    artifact: dict[str, Any]
    rows: dict[str, list[dict[str, Any]]]


def build_competition_import_plan(candidate: CompetitionCandidate) -> CompetitionImportPlan:
    asset = candidate.asset
    raw = candidate.original_template
    snapshot = {"format": FORMAT, "template": raw, "bindings": candidate.original_bindings}
    digest = hashlib.sha256(_canonical(snapshot).encode()).hexdigest()
    fqdn = "cdg.competition." + asset.asset_id
    artifact_id = _id(fqdn)
    version_id = _id(fqdn + "/" + digest)
    stage_ids = [stage.stage_id for stage in asset.stages]
    if not stage_ids or len(stage_ids) != len(set(stage_ids)):
        raise ValueError("competition intake requires unique nonempty stages")
    stage_set = set(stage_ids)
    bindings = candidate.original_bindings["bindings"]
    binding_ids = [b.get("stage_id") for b in bindings]
    if len(binding_ids) != len(set(binding_ids)) or not set(binding_ids) <= stage_set:
        raise ValueError("competition intake has duplicate or orphan bindings")
    artifact = {
        "artifact_id": artifact_id, "artifact_kind": "cdg", "fqdn": fqdn,
        "namespace_root": "cdg.competition", "namespace_path": asset.family,
        "source_symbol": asset.asset_id, "description": asset.summary,
        "source_kind": "generated", "status": "draft", "is_publishable": False,
        "verified_leaf_coverage": 0.0, "leaf_count": len(stage_ids),
        "top_level_input_arity": len(raw.get("inputs", [])),
        "top_level_output_arity": len(raw.get("outputs", [])),
    }
    rows: dict[str, list[dict[str, Any]]] = {table: [] for table in TABLE_KEYS}
    rows["artifact_versions"] = [{
        "version_id": version_id, "artifact_id": artifact_id,
        "content_hash": digest, "fingerprint": digest,
        "semver": "0.0.0+intake." + digest, "is_latest": True,
    }]
    rows["artifact_audit_evidence"] = [{
        "evidence_id": _id(version_id + "/snapshot"),
        "artifact_id": artifact_id, "version_id": version_id,
        "audit_type": "asset_integrity_check", "passed": True,
        "status": "completed", "source_kind": "automated",
        "runner_version": FORMAT, "source_revision": digest,
        "upstream_version": asset.asset_version,
        "details": {"snapshot": snapshot, "scope": "intake serialization integrity only; execution and promotion not assessed"},
    }]
    for stage in asset.stages:
        rows["artifact_cdg_nodes"].append({
            "version_id": version_id, "node_id": stage.stage_id,
            "name": stage.name, "description": stage.description,
            "concept_type": stage.concept_type.value, "status": "pending",
            "matched_primitive": "",
        })
    for edge in asset.edges:
        rows["artifact_cdg_edges"].append({
            "version_id": version_id, "source_id": edge.source_stage_id,
            "target_id": edge.target_stage_id, "output_name": edge.output_name,
            "input_name": edge.input_name,
        })
    # SQL's edge primary key omits the semantic edge attributes. Those remain
    # in the immutable snapshot; repeated structural edges are projected once.
    edge_keys = TABLE_KEYS["artifact_cdg_edges"]
    rows["artifact_cdg_edges"] = list({tuple(r[k] for k in edge_keys): r for r in rows["artifact_cdg_edges"]}.values())
    for direction, field in [("input", "inputs"), ("output", "outputs")]:
        for ordinal, port in enumerate(raw.get(field, [])):
            rows["artifact_io_specs"].append({
                "io_spec_id": _id(version_id + "/" + direction + "/" + str(ordinal)),
                "artifact_id": artifact_id, "version_id": version_id,
                "direction": direction, "name": port["name"],
                "type_desc": port.get("type_desc", "Any"),
                "constraints": port.get("constraints", ""),
                "required": port.get("required", True),
                "default_value_repr": port.get("default_value_repr", ""),
                "ordinal": ordinal,
            })
    for binding in bindings:
        target = binding.get("bound_artifact_fqdn")
        if binding.get("status") != "active" or not target:
            continue
        rows["artifact_cdg_bindings"].append({
            "binding_id": _id(version_id + "/binding/" + binding["stage_id"]),
            "version_id": version_id, "node_id": binding["stage_id"],
            "bound_artifact_fqdn": target,
            "bound_version_content_hash": binding.get("bound_version_content_hash") or "",
            "binding_confidence": binding.get("binding_confidence") or 0.0,
            "binding_source": FORMAT, "status": "active",
            "action_class": binding.get("action_class") if binding.get("action_class") in ACTIONS else None,
            "alternatives": binding.get("alternatives") or [],
            "evidence_summary": {"source_binding": binding, "intake_only": True},
        })
    return CompetitionImportPlan(artifact, rows)


def _insert(connection, table: str, row: dict[str, Any]) -> None:
    fields = list(row)
    connection.execute(
        sql.SQL("INSERT INTO public.{} ({}) VALUES ({})").format(
            sql.Identifier(table), sql.SQL(",").join(map(sql.Identifier, fields)),
            sql.SQL(",").join(sql.Placeholder() for _ in fields),
        ),
        [Jsonb(row[k]) if k in JSON_FIELDS else row[k] for k in fields],
    )


def verify_competition_import(connection, plan: CompetitionImportPlan) -> None:
    """Check every projected field and the complete versioned snapshot."""
    version_id = plan.rows["artifact_versions"][0]["version_id"]
    for table, expected_rows in plan.rows.items():
        extra_filter = sql.SQL(" AND runner_version=%s") if table == "artifact_audit_evidence" else sql.SQL("")
        params = (version_id, FORMAT) if table == "artifact_audit_evidence" else (version_id,)
        actual = connection.execute(
            sql.SQL("SELECT * FROM public.{} WHERE version_id=%s").format(sql.Identifier(table)) + extra_filter,
            params,
        ).fetchall()
        if len(actual) != len(expected_rows):
            raise ValueError("competition intake row count mismatch in " + table)
        keys = TABLE_KEYS[table]
        actual_by_key = {tuple(str(r[k]) for k in keys): r for r in actual}
        for row in expected_rows:
            found = actual_by_key.get(tuple(str(row[k]) for k in keys))
            if found is None:
                raise ValueError("competition intake row missing in " + table)
            for field, value in row.items():
                # A historical version may correctly no longer be latest.
                if field == "is_latest":
                    continue
                current = found[field]
                if field.endswith("_id") and current is not None:
                    current = str(current)
                if current != value:
                    raise ValueError("competition intake field mismatch in " + table + "." + field)


def import_competition_drafts(connection, plans: list[CompetitionImportPlan]) -> dict[str, int]:
    """Import the whole batch atomically; retain old versions and approvals.

    Requires a psycopg connection with dict_row. The caller owns credentials.
    Existing approved/flagged/withdrawn artifacts are never changed here.
    """
    counts = Counter(imported=0, unchanged=0)
    with connection.transaction():
        for plan in sorted(plans, key=lambda p: p.artifact["fqdn"]):
            artifact = plan.artifact
            connection.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s,0))", (artifact["fqdn"],))
            existing = connection.execute("SELECT artifact_id,status,is_publishable FROM artifacts WHERE fqdn=%s FOR UPDATE", (artifact["fqdn"],)).fetchone()
            if existing and str(existing["artifact_id"]) != artifact["artifact_id"]:
                raise ValueError("competition artifact identity collision")
            version_id = plan.rows["artifact_versions"][0]["version_id"]
            version = connection.execute("SELECT version_id FROM artifact_versions WHERE version_id=%s", (version_id,)).fetchone()
            if version:
                verify_competition_import(connection, plan)
                counts["unchanged"] += 1
                continue
            if existing and (existing["status"] != "draft" or existing["is_publishable"]):
                raise ValueError("draft intake cannot replace an approved or protected artifact")
            if not existing:
                _insert(connection, "artifacts", artifact)
            else:
                fields = [k for k in artifact if k not in {"artifact_id", "fqdn"}]
                connection.execute(sql.SQL("UPDATE artifacts SET {} WHERE artifact_id=%s").format(
                    sql.SQL(",").join(sql.SQL("{}=%s").format(sql.Identifier(k)) for k in fields)
                ), [artifact[k] for k in fields] + [artifact["artifact_id"]])
            connection.execute("UPDATE artifact_versions SET is_latest=false WHERE artifact_id=%s", (artifact["artifact_id"],))
            for table, rows in plan.rows.items():
                for row in rows:
                    _insert(connection, table, row)
            verify_competition_import(connection, plan)
            state = connection.execute("SELECT status,is_publishable,verified_leaf_coverage FROM artifacts WHERE artifact_id=%s", (artifact["artifact_id"],)).fetchone()
            if state != {"status": "draft", "is_publishable": False, "verified_leaf_coverage": 0.0}:
                raise ValueError("intake unexpectedly changed approval state")
            counts["imported"] += 1
    return dict(counts)

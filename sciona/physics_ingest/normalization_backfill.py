"""Evidence-preserving normalization of existing raw expression records."""

from collections import Counter
import contextlib
import io
import hashlib
from functools import lru_cache
from importlib.metadata import version, PackageNotFoundError
from pathlib import Path
from typing import Any

from psycopg.types.json import Jsonb

from sciona.physics_ingest.normalization import normalize_candidate_expression_draft


@lru_cache(maxsize=1)
def _runner_basis() -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    sources = [
        "physics_ingest/normalization_backfill.py", "physics_ingest/normalization.py",
        "ghost/symbolic.py", "ghost/symbolic_normalization.py",
    ]
    dependencies = {}
    for name in ("sympy", "antlr4-python3-runtime"):
        try:
            dependencies[name] = version(name)
        except PackageNotFoundError:
            dependencies[name] = None
    return {
        "implementation_hashes": {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in sources},
        "dependency_versions": dependencies,
    }


def prepare_normalization(row: dict[str, Any], *, reprocess_owned: bool = False) -> dict[str, Any]:
    if reprocess_owned and row.get("parse_status") in {"normalized", "parse_failed"}:
        old_evidence = row.get("evidence_json") or {}
        if old_evidence.get("normalization_backfill", {}).get("runner_version") not in {"normalization-backfill.v1", "normalization-backfill.v2"}:
            raise ValueError("only this backfill's own pending normalization can be reprocessed")
        fresh = prepare_normalization({**row, "parse_status": "raw_imported", "evidence_json": {}})
        prior = dict(old_evidence)
        history = list(prior.pop("normalization_history", []))
        history.append({"parse_status": row["parse_status"], "evidence": prior})
        fresh["evidence_json"] = {**old_evidence, **fresh["evidence_json"], "normalization_history": history}
        fresh["original_evidence"] = old_evidence
        fresh["original_parse_status"] = row["parse_status"]
        return fresh
    if row.get("parse_status") != "raw_imported" or row.get("review_status") != "needs_human":
        raise ValueError("normalization backfill requires an unprocessed review-pending expression")
    source = dict(row)
    if source.get("candidate_id"):
        source["candidate_id"] = str(source["candidate_id"])
    # Parser diagnostics can contain formulas; keep them in private evidence,
    # never in process output or aggregate reports.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        draft = normalize_candidate_expression_draft(
            source, artifact_id=str(row["artifact_id"]), version_id=str(row["version_id"]),
            expression_kind=row.get("expression_kind", "equation"),
            expression_role=row.get("expression_role", "primary"),
            require_dimensions=True,
        )
    evidence = dict(row.get("evidence_json") or {})
    generated = draft.row.evidence_json
    conflicts = [key for key in generated if key in evidence and evidence[key] != generated[key]]
    if conflicts:
        raise ValueError("normalization would overwrite existing evidence")
    evidence.update(generated)
    evidence["normalization_backfill"] = {
        "runner_version": "normalization-backfill.v2",
        **_runner_basis(),
        "scope": "parse and serialization roundtrip only; no semantic or human approval",
    }
    return {
        "expression_id": row["expression_id"], "version_id": row["version_id"],
        "original_formula": row["raw_formula"],
        "original_evidence": dict(row.get("evidence_json") or {}),
        "original_parse_status": row["parse_status"],
        "sympy_srepr": draft.row.sympy_srepr,
        "canonical_expr_hash": draft.row.canonical_expr_hash,
        "topology_hash": draft.row.topology_hash,
        "parse_status": draft.row.parse_status,
        "parse_confidence": draft.row.parse_confidence,
        "evidence_json": evidence,
        "diagnostic_codes": [d.code for d in draft.diagnostics],
    }


def apply_normalization(connection, updates: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    with connection.transaction():
        for update in updates:
            result = connection.execute(
                "UPDATE artifact_symbolic_expressions SET sympy_srepr=%s,canonical_expr_hash=%s,"
                "topology_hash=%s,parse_status=%s,parse_confidence=%s,evidence_json=%s "
                "WHERE expression_id=%s AND version_id=%s AND raw_formula=%s "
                "AND parse_status=%s AND review_status='needs_human' AND evidence_json=%s",
                (update["sympy_srepr"], update["canonical_expr_hash"], update["topology_hash"],
                 update["parse_status"], update["parse_confidence"], Jsonb(update["evidence_json"]),
                 update["expression_id"], update["version_id"], update["original_formula"], update["original_parse_status"], Jsonb(update["original_evidence"])),
            )
            if result.rowcount != 1:
                raise ValueError("expression changed during normalization; batch rolled back")
            counts[update["parse_status"]] += 1
    return dict(counts)

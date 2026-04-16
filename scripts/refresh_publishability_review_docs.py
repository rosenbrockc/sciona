from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg


REPO_ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = REPO_ROOT / "docs" / "audit"
STATUS_JSON_PATH = AUDIT_DIR / "unpublished_atom_audit_status.json"
STATUS_MD_PATH = AUDIT_DIR / "UNPUBLISHED_ATOM_AUDIT_STATUS.md"
QUEUE_JSON_PATH = AUDIT_DIR / "publishability_review_batch_queue.json"
QUEUE_MD_PATH = AUDIT_DIR / "PUBLISHABILITY_REVIEW_BATCH_QUEUE.md"

DB_URL = "postgresql://postgres:postgres@127.0.0.1:54322/postgres"
TRUST_READY = {"ready_for_manifest_merge", "ready_for_publication", "trust_ready"}
REVIEW_PASS = {"pass", "pass_with_limits"}


def _domain_from_fqdn(fqdn: str) -> str:
    parts = fqdn.split(".")
    if len(parts) >= 3 and parts[:2] == ["sciona", "atoms"]:
        return parts[2]
    return "unknown"


def _fetch_rows() -> tuple[list[dict[str, Any]], dict[str, int], int]:
    with psycopg.connect(DB_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                with rollup as (
                  select distinct on (atom_id)
                    atom_id,
                    overall_verdict,
                    review_status,
                    review_semantic_verdict,
                    review_developer_semantics_verdict,
                    trust_readiness
                  from public.atom_audit_rollups
                  order by atom_id, updated_at desc
                )
                select
                  a.fqdn,
                  a.is_publishable,
                  r.review_status,
                  r.trust_readiness,
                  r.review_semantic_verdict,
                  r.review_developer_semantics_verdict,
                  r.overall_verdict,
                  exists(select 1 from public.atom_io_specs s where s.atom_id = a.atom_id) as has_io_specs,
                  exists(select 1 from public.atom_parameters p where p.atom_id = a.atom_id) as has_parameters,
                  exists(
                    select 1 from public.atom_descriptions d
                    where d.atom_id = a.atom_id
                      and d.kind = 'dejargonized'
                      and coalesce(d.language, 'en') = 'en'
                  ) as has_description,
                  exists(select 1 from public.atom_references ar where ar.atom_id = a.atom_id) as has_references
                from public.atoms a
                left join rollup r on r.atom_id = a.atom_id
                order by a.fqdn
                """
            )
            columns = [d.name for d in cur.description]
            rows = [dict(zip(columns, row, strict=True)) for row in cur.fetchall()]

    totals = {
        "atoms": len(rows),
        "publishable_atoms": sum(1 for row in rows if row["is_publishable"]),
    }
    totals["non_publishable_atoms"] = totals["atoms"] - totals["publishable_atoms"]
    return rows, totals, totals["publishable_atoms"]


def _is_publishable_rollup(row: dict[str, Any]) -> bool:
    if row["review_status"] != "approved":
        return False
    if row["review_semantic_verdict"] not in REVIEW_PASS:
        return False
    if row["review_developer_semantics_verdict"] not in REVIEW_PASS:
        return False
    if row["trust_readiness"] not in TRUST_READY:
        return False
    return row["overall_verdict"] not in {"broken", "misleading"}


def _blockers(row: dict[str, Any]) -> list[str]:
    blockers: list[str] = []
    if not _is_publishable_rollup(row):
        blockers.append("publishable_rollup")
    if not row["has_io_specs"]:
        blockers.append("io_specs")
    if not row["has_parameters"]:
        blockers.append("parameters")
    if not row["has_description"]:
        blockers.append("description")
    if not row["has_references"]:
        blockers.append("references")
    return blockers


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    blockers = _blockers(row)
    return {
        "fqdn": row["fqdn"],
        "domain": _domain_from_fqdn(row["fqdn"]),
        "review_status": row["review_status"] or "missing_row",
        "trust_readiness": row["trust_readiness"] or "missing_row",
        "semantic_verdict": row["review_semantic_verdict"] or "missing_row",
        "developer_semantic_verdict": row["review_developer_semantics_verdict"] or "missing_row",
        "overall_verdict": row["overall_verdict"] or "missing_row",
        "blockers": blockers,
    }


def _build_status_payload(rows: list[dict[str, Any]], totals: dict[str, int], generated_at: str) -> dict[str, Any]:
    unpublished = [_normalize_row(row) for row in rows if not row["is_publishable"]]

    blocker_counts: Counter[str] = Counter()
    combo_counts: Counter[tuple[str, ...]] = Counter()
    review_counts: Counter[tuple[str, str]] = Counter()
    domain_counts: Counter[str] = Counter()
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for row in unpublished:
        for blocker in row["blockers"]:
            blocker_counts[blocker] += 1
        combo_counts[tuple(row["blockers"])] += 1
        review_counts[(row["review_status"], row["trust_readiness"])] += 1
        domain_counts[row["domain"]] += 1
        grouped[row["domain"]].append(row)

    domains = []
    for domain in sorted(grouped):
        atoms = sorted(grouped[domain], key=lambda item: item["fqdn"])
        missing = Counter()
        for atom in atoms:
            for blocker in atom["blockers"]:
                missing[blocker] += 1
        domains.append(
            {
                "domain": domain,
                "atom_count": len(atoms),
                "missing_publishable_rollup": missing["publishable_rollup"],
                "missing_io_specs": missing["io_specs"],
                "missing_parameters": missing["parameters"],
                "missing_description": missing["description"],
                "missing_references": missing["references"],
                "atoms": atoms,
            }
        )

    return {
        "generated_at": generated_at,
        "source": "local_supabase",
        "totals": totals,
        "blocker_counts": dict(sorted(blocker_counts.items())),
        "top_blocker_combinations": [
            {"blockers": list(combo), "atom_count": count}
            for combo, count in combo_counts.most_common(10)
        ],
        "top_review_statuses": [
            {"review_status": review_status, "trust_readiness": trust_readiness, "atom_count": count}
            for (review_status, trust_readiness), count in review_counts.most_common(10)
        ],
        "domains": domains,
        "largest_non_publishable_domains": [
            {"domain": domain, "atom_count": count}
            for domain, count in domain_counts.most_common()
        ],
    }


def _render_status_md(payload: dict[str, Any]) -> str:
    lines = [
        "# Unpublished Atom Audit Status",
        "",
        f"Generated from the live local Supabase replay on {payload['generated_at']}.",
        "",
        "This document is a working debt register for every currently unpublished atom.",
        "",
        "## Summary",
        "",
        f"- Total atoms in local catalog: `{payload['totals']['atoms']}`",
        f"- Publishable atoms: `{payload['totals']['publishable_atoms']}`",
        f"- Non-publishable atoms: `{payload['totals']['non_publishable_atoms']}`",
        "",
        "### Marginal Blocker Counts",
        "",
    ]
    for blocker, count in payload["blocker_counts"].items():
        lines.append(f"- `{blocker}`: `{count}`")
    lines.extend(["", "### Top Exact Blocker Combinations", ""])
    for item in payload["top_blocker_combinations"]:
        combo = ",".join(item["blockers"]) if item["blockers"] else "none"
        lines.append(f"- `{combo}`: `{item['atom_count']}`")
    lines.extend(["", "### Largest Non-Publishable Domains", ""])
    for item in payload["largest_non_publishable_domains"]:
        lines.append(f"- `{item['domain']}`: `{item['atom_count']}`")
    lines.extend(
        [
            "",
            "## Status Legend",
            "",
            "- `publishable_rollup`: no approved audit rollup satisfying the current publication rule",
            "- `io_specs`: no atom IO spec rows",
            "- `parameters`: no atom parameter rows",
            "- `description`: no English low-jargon description",
            "- `references`: no atom references rows",
            "- `missing_row`: there is no audit rollup row for the atom yet",
        ]
    )

    for domain in payload["domains"]:
        lines.extend(
            [
                "",
                f"## {domain['domain']}",
                "",
                f"- Non-publishable atoms: `{domain['atom_count']}`",
                f"- Missing publishable rollup: `{domain['missing_publishable_rollup']}`",
                f"- Missing IO specs: `{domain['missing_io_specs']}`",
                f"- Missing parameters: `{domain['missing_parameters']}`",
                f"- Missing description: `{domain['missing_description']}`",
                f"- Missing references: `{domain['missing_references']}`",
                "",
                "| Atom | Review | Trust | Semantic | Dev Semantic | Verdict | Blockers |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for atom in domain["atoms"]:
            blockers = ", ".join(atom["blockers"])
            lines.append(
                f"| `{atom['fqdn']}` | `{atom['review_status']}` | `{atom['trust_readiness']}` | "
                f"`{atom['semantic_verdict']}` | `{atom['developer_semantic_verdict']}` | "
                f"`{atom['overall_verdict']}` | `{blockers}` |"
            )
    return "\n".join(lines) + "\n"


def _refresh_queue(
    unpublished_fqdns: set[str],
    generated_at: str,
    blocker_counts: dict[str, int],
) -> tuple[dict[str, Any], str]:
    queue = json.loads(QUEUE_JSON_PATH.read_text())
    new_batches = []
    repo_counts: Counter[str] = Counter()
    wave_counts: Counter[str] = Counter()
    special_slices = queue.get("special_slices", {})

    for batch in queue["batches"]:
        atoms = [atom for atom in batch["atoms"] if atom["fqdn"] in unpublished_fqdns]
        if not atoms:
            continue
        new_batch = dict(batch)
        new_batch["atoms"] = atoms
        new_batch["atom_count"] = len(atoms)
        new_batches.append(new_batch)
        repo_counts[new_batch["repo_owner"]] += len(atoms)
        wave_counts[new_batch["recommended_wave"]] += len(atoms)

    filtered_special_slices: dict[str, Any] = {}
    for key, value in special_slices.items():
        if isinstance(value, dict) and isinstance(value.get("atoms"), list):
            kept_atoms = [fqdn for fqdn in value["atoms"] if fqdn in unpublished_fqdns]
            filtered_special_slices[key] = {
                **value,
                "atoms": kept_atoms,
                "atom_count": len(kept_atoms),
            }
        else:
            filtered_special_slices[key] = value

    refreshed = {
        "generated_at": generated_at,
        "source": "local_supabase_filtered_existing_queue",
        "batch_count": len(new_batches),
        "batches": new_batches,
        "totals": {
            "non_publishable_atoms": len(unpublished_fqdns),
        },
        "repo_counts": dict(sorted(repo_counts.items())),
        "wave_counts": dict(sorted(wave_counts.items())),
        "blocker_counts": blocker_counts,
        "special_slices": filtered_special_slices,
    }

    lines = [
        "# Publishability Review Batch Queue",
        "",
        f"Generated from `docs/audit/unpublished_atom_audit_status.json` on {generated_at}.",
        "",
        f"- Remaining unpublished atoms: `{len(unpublished_fqdns)}`",
        f"- Remaining worker batches: `{len(new_batches)}`",
        "",
        "## Batches",
        "",
    ]
    for batch in new_batches:
        reps = ", ".join(f"`{fqdn}`" for fqdn in batch["representative_atoms"][:3])
        lines.extend(
            [
                f"### {batch['batch_id']}",
                "",
                f"- Repo: `{batch['repo_owner']}`",
                f"- Wave: `{batch['recommended_wave']}`",
                f"- Atoms: `{batch['atom_count']}`",
                f"- Blocker class: `{batch['blocker_class']}`",
                f"- Primary blocker pattern: `{batch['primary_blocker_pattern']}`",
                f"- Representative atoms: {reps}",
                "",
            ]
        )
    lines.append(f"- The canonical machine-readable queue is [publishability_review_batch_queue.json]({QUEUE_JSON_PATH}).")
    return refreshed, "\n".join(lines) + "\n"


def main() -> int:
    generated_at = datetime.now(timezone.utc).isoformat()
    rows, totals, _ = _fetch_rows()
    payload = _build_status_payload(rows, totals, generated_at)
    STATUS_JSON_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    STATUS_MD_PATH.write_text(_render_status_md(payload))

    unpublished_fqdns = {
        atom["fqdn"]
        for domain in payload["domains"]
        for atom in domain["atoms"]
    }
    queue_payload, queue_md = _refresh_queue(
        unpublished_fqdns,
        generated_at,
        payload["blocker_counts"],
    )
    QUEUE_JSON_PATH.write_text(json.dumps(queue_payload, indent=2) + "\n")
    QUEUE_MD_PATH.write_text(queue_md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

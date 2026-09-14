#!/usr/bin/env python3
"""Normalize raw physics expressions without changing validation or review states."""

import argparse
from collections import Counter
import json
import os

import psycopg
from psycopg.rows import dict_row

from sciona.physics_ingest.normalization_backfill import prepare_normalization, apply_normalization


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--reprocess-owned", action="store_true", help="Reparse this backfill's pending records, preserving prior evidence")
    parser.add_argument("--only-failed", action="store_true", help="Restrict reprocessing to parse failures")
    args = parser.parse_args()
    if args.limit < 1:
        parser.error("limit must be positive")
    if args.only_failed and not args.reprocess_owned:
        parser.error("--only-failed requires --reprocess-owned")
    url = os.environ.get("SCIONA_DATA_CATALOG_DATABASE_URL")
    if not url:
        parser.error("Set SCIONA_DATA_CATALOG_DATABASE_URL in the runtime environment")
    with psycopg.connect(url, row_factory=dict_row, options="-c default_transaction_read_only=on") as db:
        rows = db.execute(
            "SELECT e.*,q.source_payload FROM artifact_symbolic_expressions e "
            "LEFT JOIN physics_equation_candidates q USING(candidate_id) "
            "WHERE e.review_status='needs_human' AND (e.parse_status='raw_imported' OR "
            "(%s AND e.parse_status IN ('normalized','parse_failed') AND "
            "e.evidence_json->'normalization_backfill'->>'runner_version' IN ('normalization-backfill.v1','normalization-backfill.v2'))) "
            "AND (NOT %s OR e.parse_status='parse_failed') "
            "ORDER BY e.expression_id LIMIT %s", (args.reprocess_owned, args.only_failed, args.limit),
        ).fetchall()
    updates = []
    failures = Counter()
    diagnostics = Counter()
    for index, row in enumerate(rows, 1):
        try:
            update = prepare_normalization(row, reprocess_owned=args.reprocess_owned)
            updates.append(update)
            diagnostics.update(update["diagnostic_codes"])
        except Exception as exc:
            # Leave failed preparations untouched. Do not print formulas or
            # exception messages, which can contain source records.
            failures[type(exc).__name__] += 1
        if index % 50 == 0:
            print(json.dumps({"processed": index, "total": len(rows)}), flush=True)
    report = {
        "examined": len(rows), "prepared": len(updates), "preparation_failures": dict(failures),
        "parse_results": dict(Counter(u["parse_status"] for u in updates)),
        "diagnostic_codes": dict(diagnostics), "applied": False,
    }
    if args.apply and updates:
        with psycopg.connect(url, row_factory=dict_row, autocommit=True) as db:
            report["applied_counts"] = apply_normalization(db, updates)
        report["applied"] = True
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

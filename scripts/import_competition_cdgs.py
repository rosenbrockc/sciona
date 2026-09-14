#!/usr/bin/env python3
"""Plan or atomically import competition CDGs as unapproved draft versions."""

import argparse
from collections import Counter
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from sciona.competition_import import build_competition_import_plan, import_competition_drafts
from sciona.competition_promotion import load_competition_candidates


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    candidates = load_competition_candidates(args.competition_dir)
    plans = [build_competition_import_plan(c) for c in candidates]
    row_counts = Counter()
    for plan in plans:
        row_counts.update({table: len(rows) for table, rows in plan.rows.items()})
    report = {"templates": len(plans), "projected_rows": dict(row_counts), "applied": False}
    if args.apply:
        url = os.environ.get("SCIONA_DATA_CATALOG_DATABASE_URL")
        if not url:
            parser.error("Set SCIONA_DATA_CATALOG_DATABASE_URL in the runtime environment")
        with psycopg.connect(url, autocommit=True, row_factory=dict_row, connect_timeout=10) as connection:
            report["result"] = import_competition_drafts(connection, plans)
        report["applied"] = True
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

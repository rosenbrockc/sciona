#!/usr/bin/env python3
"""Read-only promotion inventory; emits aggregate counts, never source payloads."""

import argparse
import json
import os
from pathlib import Path

import psycopg
from psycopg.rows import dict_row

from sciona.competition_promotion import load_competition_candidates, summarize_competition_intake


def audit(database_url: str, competition_dir: Path) -> dict:
    candidates = load_competition_candidates(competition_dir)
    with psycopg.connect(
        database_url, row_factory=dict_row, connect_timeout=10,
        options="-c default_transaction_read_only=on -c statement_timeout=30000",
    ) as connection:
        catalog = {}
        for table, version_table, id_column in [
            ("artifacts", "artifact_versions", "artifact_id"),
            ("atoms", "atom_versions", "atom_id"),
        ]:
            # Identifiers are fixed application constants, not user inputs.
            rows = connection.execute(
                f"SELECT a.fqdn,a.status,a.is_publishable,v.content_hash FROM {table} a "
                f"LEFT JOIN {version_table} v ON a.{id_column}=v.{id_column} AND v.is_latest"
            ).fetchall()
            catalog.update({r["fqdn"]: r for r in rows})
        inventory = connection.execute(
            "SELECT source_kind,status,is_publishable,count(*) AS count FROM artifacts "
            "WHERE artifact_kind='cdg' GROUP BY 1,2,3 ORDER BY 1,2,3"
        ).fetchall()
        identities = connection.execute(
            "SELECT fqdn,source_symbol FROM artifacts WHERE artifact_kind='cdg'"
        ).fetchall()
        imported = sum(
            any(c.asset.asset_id == r["source_symbol"] or r["fqdn"] in
                {c.asset.asset_id, 'cdg.competition.' + c.asset.asset_id}
                for r in identities)
            for c in candidates
        )
        physics = connection.execute(
            "SELECT p.publication_ready,count(*) AS count "
            "FROM physics_cdg_artifact_envelope_publication p "
            "JOIN artifacts a USING(artifact_id) WHERE a.source_kind='generated' "
            "AND (a.namespace_root LIKE '%physics%' OR a.source_package LIKE '%physics%') "
            "GROUP BY 1"
        ).fetchall()
    return {
        "read_only": True,
        "competition": {**summarize_competition_intake(candidates, catalog), "represented_in_catalog": imported},
        "catalog_cdg_inventory": inventory,
        "physics_publication_readiness": physics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--competition-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    url = os.environ.get("SCIONA_DATA_CATALOG_DATABASE_URL")
    if not url:
        parser.error("Set SCIONA_DATA_CATALOG_DATABASE_URL in the runtime environment")
    report = audit(url, args.competition_dir)
    text = json.dumps(report, indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()

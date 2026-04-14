#!/usr/bin/env python3
"""Sync local skeleton-family CDG artifacts into Supabase and Memgraph."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging

from sciona.config import AgeomConfig
from sciona.graph_store import GraphStore
from sciona.services.skeleton_catalog_sync import (
    load_skeleton_artifact_bundles,
    sync_bundle_to_supabase,
    sync_bundles_to_graph_store,
)

log = logging.getLogger(__name__)


def _create_supabase_client():
    from scripts.backfill_utils import create_supabase_client_from_env

    return create_supabase_client_from_env()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asset-id",
        action="append",
        default=[],
        help="Restrict sync to one or more skeleton asset ids.",
    )
    parser.add_argument(
        "--skip-memgraph",
        action="store_true",
        help="Only sync Supabase artifact tables.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the deterministic payload summary without writing anything.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit a machine-readable summary.",
    )
    return parser.parse_args()


async def _sync_memgraph(config: AgeomConfig, bundles):
    async with GraphStore(
        uri=config.memgraph_uri,
        user=config.memgraph_user,
        password=config.memgraph_password,
    ) as graph_store:
        return await sync_bundles_to_graph_store(graph_store, bundles)


def main() -> int:
    args = parse_args()
    bundles = load_skeleton_artifact_bundles()
    selected_ids = {str(asset_id).strip() for asset_id in args.asset_id if str(asset_id).strip()}
    if selected_ids:
        bundles = [bundle for bundle in bundles if bundle.asset_id in selected_ids]

    payload = {
        "asset_ids": [bundle.asset_id for bundle in bundles],
        "artifacts": [
            {
                "asset_id": bundle.asset_id,
                "fqdn": bundle.artifact["fqdn"],
                "artifact_id": bundle.artifact["artifact_id"],
                "version_id": bundle.version["version_id"],
                "semver": bundle.version["semver"],
                "content_hash": bundle.version["content_hash"],
                "is_publishable": bool(bundle.artifact["is_publishable"]),
                "topo_hash": bundle.artifact["topo_hash"],
                "node_count": len(bundle.cdg_nodes),
                "edge_count": len(bundle.cdg_edges),
                "reference_count": len(bundle.references),
            }
            for bundle in bundles
        ],
    }
    if args.dry_run:
        if args.json:
            print(json.dumps(payload, indent=2, sort_keys=True))
        else:
            for row in payload["artifacts"]:
                print(
                    f"{row['asset_id']}: fqdn={row['fqdn']} semver={row['semver']} "
                    f"publishable={row['is_publishable']} nodes={row['node_count']} edges={row['edge_count']}"
                )
        return 0

    supabase = _create_supabase_client()
    for bundle in bundles:
        sync_bundle_to_supabase(supabase, bundle)

    memgraph_counts: list[tuple[str, dict[str, int]]] = []
    if not args.skip_memgraph and bundles:
        memgraph_counts = asyncio.run(_sync_memgraph(AgeomConfig(), bundles))

    payload["memgraph"] = [
        {"asset_id": asset_id, **counts} for asset_id, counts in memgraph_counts
    ]
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        log.info("Synced %d skeleton artifact(s) to Supabase", len(bundles))
        if memgraph_counts:
            log.info("Synced %d skeleton artifact projection(s) to Memgraph", len(memgraph_counts))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    raise SystemExit(main())

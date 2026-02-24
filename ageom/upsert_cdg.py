"""Async entrypoint for upserting CDGs into Neo4j."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ageom.config import AgeomConfig
from ageom.graph_store import (
    Neo4jStore,
    extract_contract_metadata,
    extract_witness_metadata,
)


async def upsert_repo(
    repo_path: Path, repo_name: str, config: AgeomConfig
) -> dict[str, Any]:
    """Discover and upsert all CDG JSON files in *repo_path*.

    Returns a summary dict: ``{cdg_file: counts_dict, ...}``.
    """
    cdg_files = sorted(repo_path.glob("*cdg*.json"))
    if not cdg_files:
        print(f"  No CDG files found in {repo_path}")
        return {}

    # Collect all node_ids across CDGs for metadata extraction
    all_node_ids: list[str] = []
    cdg_data: list[tuple[Path, dict[str, Any]]] = []
    for cf in cdg_files:
        with open(cf) as f:
            data = json.load(f)
        cdg_data.append((cf, data))
        for node in data.get("nodes", []):
            all_node_ids.append(node["node_id"])

    # Extract witness & contract metadata from sibling .py files
    witness_meta = extract_witness_metadata(repo_path, all_node_ids)
    contract_meta = extract_contract_metadata(repo_path, all_node_ids)

    summary: dict[str, Any] = {}
    async with Neo4jStore(
        uri=config.neo4j_uri,
        user=config.neo4j_user,
        password=config.neo4j_password,
    ) as store:
        await store.ensure_constraints()

        for cf, data in cdg_data:
            print(f"  Upserting {cf.name} ...")
            counts = await store.upsert_cdg(
                repo=repo_name,
                cdg_dict=data,
                witness_meta=witness_meta,
                contract_meta=contract_meta,
            )
            summary[cf.name] = counts
            print(
                f"    atoms={counts['atoms']} "
                f"ports={counts['input_ports']}in/{counts['output_ports']}out "
                f"edges={counts['data_flow']}df/{counts['parent_of']}parent "
                f"deleted={counts['deleted']}"
            )

    return summary

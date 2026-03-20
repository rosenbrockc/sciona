"""Async entrypoint for upserting CDGs into Memgraph."""

from __future__ import annotations

import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any

from sciona.config import AgeomConfig
from sciona.graph_store import (
    GraphStore,
    extract_contract_metadata,
    extract_witness_metadata,
)

_logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CDG sanitization (run before upsert)
# ---------------------------------------------------------------------------

def _dedup_nodes(cdg: dict[str, Any]) -> dict[str, Any]:
    """Merge duplicate ``node_id`` entries in a CDG.

    Some ingested CDGs contain multiple nodes sharing the same ``node_id``
    (e.g. overloaded Julia/C++ constructors).  When the graph store MERGEs
    on ``fqn = repo.node_id``, the last writer's scalar properties win
    while input/output ports from *all* writers accumulate, causing
    ``n_inputs`` to diverge from the actual HAS_INPUT count.

    Strategy: keep the entry with the most inputs (richest signature) and
    merge the extra inputs/outputs from siblings as a union by port name.
    """
    nodes: list[dict[str, Any]] = cdg.get("nodes", [])
    id_counts = Counter(n["node_id"] for n in nodes)
    dupes = {nid for nid, cnt in id_counts.items() if cnt > 1}
    if not dupes:
        return cdg

    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []

    for node in nodes:
        nid = node["node_id"]
        if nid not in dupes:
            if nid not in merged:
                merged[nid] = node
                order.append(nid)
            continue

        if nid not in merged:
            merged[nid] = dict(node)  # shallow copy
            merged[nid]["inputs"] = list(node.get("inputs") or [])
            merged[nid]["outputs"] = list(node.get("outputs") or [])
            order.append(nid)
        else:
            # Merge: keep the richer entry as base, union ports by name
            existing = merged[nid]
            # If this variant has more inputs, adopt its scalar props
            if len(node.get("inputs") or []) > len(existing.get("inputs") or []):
                saved_inputs = existing["inputs"]
                saved_outputs = existing["outputs"]
                existing.update(node)
                existing["inputs"] = list(node.get("inputs") or [])
                existing["outputs"] = list(node.get("outputs") or [])
                # Merge back old ports not present in the new base
                extra_in = node.get("inputs") or []
                extra_out = node.get("outputs") or []
            else:
                extra_in = node.get("inputs") or []
                extra_out = node.get("outputs") or []

            seen_in = {p["name"] for p in existing["inputs"]}
            for p in extra_in:
                if p["name"] not in seen_in:
                    existing["inputs"].append(p)
                    seen_in.add(p["name"])

            seen_out = {p["name"] for p in existing["outputs"]}
            for p in extra_out:
                if p["name"] not in seen_out:
                    existing["outputs"].append(p)
                    seen_out.add(p["name"])

    _logger.info(
        "Deduped %d node_id(s): %s",
        len(dupes),
        ", ".join(sorted(dupes)),
    )
    return {**cdg, "nodes": [merged[nid] for nid in order]}


def _fix_childless_decomposed(cdg: dict[str, Any]) -> dict[str, Any]:
    """Downgrade decomposed nodes that have no children to atomic.

    Some CDGs have a single root marked ``decomposed`` with an empty
    ``children`` list and zero edges.  These are invisible to the
    isomorphism search layers and should be ``atomic`` instead.
    """
    nodes = cdg.get("nodes", [])
    edges = cdg.get("edges", [])
    all_ids = {n["node_id"] for n in nodes}
    parent_ids_from_edges = {e.get("source_id") for e in edges}

    changed = False
    new_nodes = []
    for node in nodes:
        children = node.get("children", [])
        # Only count children that actually exist in this CDG
        real_children = [c for c in children if c in all_ids]
        has_edge_children = node["node_id"] in parent_ids_from_edges

        if (
            node.get("status") == "decomposed"
            and not real_children
            and not has_edge_children
        ):
            node = dict(node)
            node["status"] = "atomic"
            node["children"] = []
            changed = True
            _logger.info(
                "Downgraded childless decomposed node '%s' to atomic",
                node.get("name", node["node_id"]),
            )
        new_nodes.append(node)

    if changed:
        return {**cdg, "nodes": new_nodes}
    return cdg


def _normalize_edge_keys(cdg: dict[str, Any]) -> dict[str, Any]:
    """Normalise edge key names to ``source_id``/``target_id``.

    Some CDGs (e.g. from the Bayesian / conjugate-priors ingester) use
    ``source``/``target`` instead.  Map them so downstream code that
    expects ``source_id``/``target_id`` works uniformly.
    """
    edges = cdg.get("edges", [])
    if not edges:
        return cdg

    # Check if normalisation is needed
    sample = edges[0]
    if "source_id" in sample:
        return cdg  # already normalised

    new_edges = []
    for e in edges:
        ne = dict(e)
        if "source" in ne and "source_id" not in ne:
            ne["source_id"] = ne.pop("source")
        if "target" in ne and "target_id" not in ne:
            ne["target_id"] = ne.pop("target")
        new_edges.append(ne)

    return {**cdg, "edges": new_edges}


def sanitize_cdg(cdg: dict[str, Any]) -> dict[str, Any]:
    """Apply all pre-upsert sanitization passes to a CDG dict."""
    cdg = _normalize_edge_keys(cdg)
    cdg = _dedup_nodes(cdg)
    cdg = _fix_childless_decomposed(cdg)
    return cdg


# ---------------------------------------------------------------------------
# Repo upsert
# ---------------------------------------------------------------------------

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
        data = sanitize_cdg(data)
        cdg_data.append((cf, data))
        for node in data.get("nodes", []):
            all_node_ids.append(node["node_id"])

    # Extract witness & contract metadata from sibling .py files
    witness_meta = extract_witness_metadata(repo_path, all_node_ids)
    contract_meta = extract_contract_metadata(repo_path, all_node_ids)

    summary: dict[str, Any] = {}
    async with GraphStore(
        uri=config.memgraph_uri,
        user=config.memgraph_user,
        password=config.memgraph_password,
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

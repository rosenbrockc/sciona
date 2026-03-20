"""Structural summaries for Principal trials."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sciona.architect.handoff import CDGExport
from sciona.architect.models import NodeStatus
from sciona.graph_store import _topo_hash
from sciona.principal.structure_objective import compute_structure_loss
from sciona.synthesizer.ghost_sim import GhostSimReport
from sciona.types import MatchResult


def summarize_trial_structure(
    cdg: CDGExport,
    *,
    ghost_report: GhostSimReport | None = None,
    match_results: list[MatchResult] | None = None,
) -> dict[str, Any]:
    """Summarize a CDG structure for trial-history comparisons."""
    nodes = list(cdg.nodes)
    edges = list(cdg.edges)
    atomic_nodes = [node for node in nodes if node.status == NodeStatus.ATOMIC]
    successful = {
        result.pdg_node.predicate_id
        for result in (match_results or [])
        if result.success
    }
    matched_leaf_count = sum(1 for node in atomic_nodes if node.node_id in successful)
    atomic_leaf_count = len(atomic_nodes)
    root_id = _infer_root_id(nodes, edges)
    topo_hash = ""
    if root_id is not None:
        topo_hash = _topo_hash(
            [node.model_dump() for node in nodes],
            [edge.model_dump() for edge in edges],
            root_id,
        )
    primitive_map = {
        node.node_id: str(node.matched_primitive or "")
        for node in atomic_nodes
        if node.matched_primitive
    }
    primitive_signature = hashlib.sha256(
        json.dumps(sorted(primitive_map.items()), separators=(",", ":")).encode()
    ).hexdigest()[:16]
    coverage = (
        float(matched_leaf_count) / float(atomic_leaf_count)
        if atomic_leaf_count > 0
        else 0.0
    )
    summary = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "atomic_leaf_count": atomic_leaf_count,
        "matched_leaf_count": matched_leaf_count,
        "verified_leaf_coverage": coverage,
        "topo_hash": topo_hash,
        "primitive_signature": primitive_signature,
        "atomic_primitives": primitive_map,
    }
    if ghost_report is not None:
        summary["ghost_coverage"] = float(ghost_report.coverage)
        summary["structure_loss"] = float(compute_structure_loss(ghost_report))
        summary["ghost_passed"] = bool(ghost_report.passed)
    return summary


def _infer_root_id(nodes: list[Any], edges: list[Any]) -> str | None:
    node_ids = {node.node_id for node in nodes}
    child_ids = {edge.target_id for edge in edges if edge.target_id in node_ids}
    roots = [node.node_id for node in nodes if node.node_id not in child_ids]
    if roots:
        return sorted(roots)[0]
    return nodes[0].node_id if nodes else None

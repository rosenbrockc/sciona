"""Structural summaries for Principal trials."""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any

from sciona.atom_identity import logical_atom_id_from_fqdn, known_atom_package_prefixes
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
    catalog: Any | None = None,
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
    concept_types = sorted({node.concept_type.value for node in atomic_nodes})
    primitive_families = {
        node.node_id: _primitive_family(str(node.matched_primitive or ""), catalog)
        for node in atomic_nodes
        if node.matched_primitive
    }
    family_counts: dict[str, int] = {}
    for family in primitive_families.values():
        family_counts[family] = family_counts.get(family, 0) + 1
    total_family_nodes = sum(family_counts.values())
    family_entropy = 0.0
    if total_family_nodes > 0:
        for count in family_counts.values():
            p = count / total_family_nodes
            family_entropy -= p * math.log2(p)
    cross_family_edge_count = 0
    cross_family_nodes: set[str] = set()
    atomic_ids = {node.node_id for node in atomic_nodes}
    concept_type_by_node = {node.node_id: node.concept_type.value for node in atomic_nodes}
    for edge in edges:
        if edge.source_id not in atomic_ids or edge.target_id not in atomic_ids:
            continue
        source_family = primitive_families.get(edge.source_id, "")
        target_family = primitive_families.get(edge.target_id, "")
        source_concept = concept_type_by_node.get(edge.source_id, "")
        target_concept = concept_type_by_node.get(edge.target_id, "")
        if (
            source_family
            and target_family
            and source_family != target_family
        ) or (source_concept and target_concept and source_concept != target_concept):
            cross_family_edge_count += 1
            cross_family_nodes.add(edge.source_id)
            cross_family_nodes.add(edge.target_id)
    foreign_family_bindings: list[str] = []
    if catalog is not None:
        for node in atomic_nodes:
            primitive_name = str(node.matched_primitive or "")
            if not primitive_name:
                continue
            prim = catalog.get(primitive_name)
            if prim is None:
                continue
            if prim.category != node.concept_type:
                foreign_family_bindings.append(node.node_id)
    summary.update(
        {
            "distinct_concept_types": concept_types,
            "distinct_concept_type_count": len(concept_types),
            "distinct_primitive_families": sorted(family_counts),
            "distinct_primitive_family_count": len(family_counts),
            "family_entropy": family_entropy,
            "cross_family_edge_count": cross_family_edge_count,
            "cross_family_node_count": len(cross_family_nodes),
            "foreign_family_binding_count": len(foreign_family_bindings),
            "foreign_family_bindings": sorted(foreign_family_bindings),
        }
    )
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


def _recognized_atom_package_prefix(label: str) -> str:
    text = str(label or "").strip()
    if not text:
        return ""
    for prefix in sorted(known_atom_package_prefixes(), key=len, reverse=True):
        if text == prefix or text.startswith(prefix + "."):
            return prefix
    return ""


def _fallback_atom_family(label: str) -> str:
    prefix = _recognized_atom_package_prefix(label)
    if not prefix:
        return ""
    logical_name = logical_atom_id_from_fqdn(label)
    logical_parts = [part for part in logical_name.split(".") if part]
    if not logical_parts:
        return prefix
    return ".".join([prefix, *logical_parts[:2]])


def _primitive_family(primitive_name: str, catalog: Any | None) -> str:
    primitive_name = str(primitive_name or "").strip()
    if not primitive_name:
        return ""
    if catalog is not None:
        prim = catalog.get(primitive_name)
        if prim is not None:
            source = str(getattr(prim, "source", "") or "").strip()
            if source:
                return source
    atom_family = _fallback_atom_family(primitive_name)
    if atom_family:
        return atom_family
    if "." in primitive_name:
        return primitive_name.split(".", 1)[0]
    return "builtin"

"""Deterministic helpers for published CDG projection payloads."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Mapping

from sciona.architect.handoff import CDGExport


def _topo_hash(nodes: list[dict[str, Any]], edges: list[dict[str, Any]], root_id: str) -> str:
    children = [node for node in nodes if node.get("parent_id") == root_id]
    child_ids = {child["node_id"] for child in children}
    if not child_ids:
        child_ids = {str(node.get("node_id", "")) for node in nodes if str(node.get("node_id", "")).strip()}
        sibling_edges = [
            edge
            for edge in edges
            if edge.get("source_id") in child_ids and edge.get("target_id") in child_ids
        ]
    else:
        sibling_edges = [
            edge
            for edge in edges
            if edge.get("source_id") in child_ids and edge.get("target_id") in child_ids
        ]
    degree_seq: list[tuple[int, int]] = []
    for child_id in sorted(child_ids):
        in_deg = sum(1 for edge in sibling_edges if edge.get("target_id") == child_id)
        out_deg = sum(1 for edge in sibling_edges if edge.get("source_id") == child_id)
        degree_seq.append((in_deg, out_deg))
    raw = str(sorted(degree_seq))
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@dataclass(frozen=True)
class PublishedCDGProjection:
    artifact_id: str
    artifact_version_id: str
    fqdn: str
    semver: str
    content_hash: str
    artifact_kind: str = "cdg"
    repo: str = ""
    source_commit: str = ""
    published_at: str = ""
    topo_hash: str = ""
    verified_leaf_coverage: float = 0.0
    n_inputs: int = 0
    n_outputs: int = 0
    nodes: list[dict[str, Any]] = field(default_factory=list)
    edges: list[dict[str, Any]] = field(default_factory=list)


def normalize_cdg_export(cdg: CDGExport | Mapping[str, Any]) -> dict[str, Any]:
    """Normalize a CDG export or mapping to plain node/edge dictionaries."""
    if isinstance(cdg, CDGExport):
        return cdg.model_dump(mode="json")
    return {
        "nodes": [
            node.model_dump(mode="json") if hasattr(node, "model_dump") else dict(node)
            for node in list(cdg.get("nodes", []) or [])
        ],
        "edges": [
            edge.model_dump(mode="json") if hasattr(edge, "model_dump") else dict(edge)
            for edge in list(cdg.get("edges", []) or [])
        ],
        "metadata": dict(cdg.get("metadata", {}) or {}),
    }


def build_published_cdg_projection(
    *,
    artifact: Mapping[str, Any],
    version: Mapping[str, Any],
    cdg: CDGExport | Mapping[str, Any],
) -> PublishedCDGProjection:
    """Build a deterministic Memgraph projection payload for a published CDG."""
    normalized = normalize_cdg_export(cdg)
    nodes = list(normalized.get("nodes", []) or [])
    edges = list(normalized.get("edges", []) or [])
    metadata = dict(normalized.get("metadata", {}) or {})
    root_id = ""
    for node in nodes:
        if not node.get("parent_id"):
            root_id = str(node.get("node_id", ""))
            break
    if not root_id and nodes:
        root_id = str(nodes[0].get("node_id", ""))
    topo_hash = _topo_hash(nodes, edges, root_id) if root_id else ""
    root_node = next((node for node in nodes if str(node.get("node_id", "")) == root_id), {})
    return PublishedCDGProjection(
        artifact_id=str(artifact.get("artifact_id", "")),
        artifact_version_id=str(version.get("version_id") or version.get("artifact_version_id", "")),
        fqdn=str(artifact.get("fqdn", "")),
        semver=str(version.get("semver", "")),
        content_hash=str(version.get("content_hash", "")),
        artifact_kind=str(artifact.get("artifact_kind", "cdg")),
        repo=str(artifact.get("repo", "") or artifact.get("namespace_root", "")),
        source_commit=str(version.get("source_commit", "")),
        published_at=str(version.get("created_at", "")),
        topo_hash=topo_hash,
        verified_leaf_coverage=float(metadata.get("verified_leaf_coverage", 0.0) or 0.0),
        n_inputs=len(root_node.get("inputs", []) or []),
        n_outputs=len(root_node.get("outputs", []) or []),
        nodes=nodes,
        edges=edges,
    )

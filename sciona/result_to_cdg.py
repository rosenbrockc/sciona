"""Convert an OrchestratorResult into a sanitize_cdg-compatible dict.

Serializes the final CDG, enriches nodes with matched primitive info,
computes verified leaf coverage, and attaches provenance metadata to the
root node.
"""

from __future__ import annotations

from dataclasses import dataclass

from sciona.architect.models import NodeStatus
from sciona.orchestrator import OrchestratorResult
from sciona.upsert_cdg import sanitize_cdg


@dataclass
class RunCDGMetadata:
    """Provenance metadata attached to the CDG root node."""

    run_id: str
    goal: str
    execution_path: str  # e.g. "verified", "structured", "rapid"
    timestamp: str  # ISO format
    verified_leaf_coverage: float  # 0.0-1.0


def orchestrator_result_to_cdg(
    result: OrchestratorResult,
    metadata: RunCDGMetadata,
) -> dict:
    """Convert an OrchestratorResult into a dict compatible with sanitize_cdg.

    Steps:
    1. Serialize nodes and edges from the CDG.
    2. Enrich matched nodes with ``matched_primitive``.
    3. Compute verified leaf coverage and store in metadata.
    4. Attach provenance metadata to the root node.
    5. Return a sanitized dict with "nodes" and "edges" keys.
    """
    cdg = result.cdg

    # Build a lookup: predicate_id -> declaration name for successful matches
    matched_primitives: dict[str, str] = {}
    for mr in result.match_results:
        if mr.success and mr.verified_match is not None:
            matched_primitives[mr.pdg_node.predicate_id] = (
                mr.verified_match.candidate.declaration.name
            )

    # Serialize nodes
    serialized_nodes: list[dict] = []
    for node in cdg.nodes:
        node_dict = node.model_dump()

        # Enrich with matched primitive if this node was successfully matched
        if node.node_id in matched_primitives:
            node_dict["matched_primitive"] = matched_primitives[node.node_id]

        serialized_nodes.append(node_dict)

    # Serialize edges
    serialized_edges: list[dict] = [edge.model_dump() for edge in cdg.edges]

    # Compute verified leaf coverage
    atomic_nodes = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
    total_atomic = len(atomic_nodes)
    matched_atomic = sum(
        1 for n in atomic_nodes if n.node_id in matched_primitives
    )
    coverage = matched_atomic / total_atomic if total_atomic > 0 else 0.0
    metadata.verified_leaf_coverage = coverage

    # Attach provenance metadata to root node (parent_id is None)
    for node_dict in serialized_nodes:
        if node_dict.get("parent_id") is None:
            node_dict["provenance"] = {
                "run_id": metadata.run_id,
                "goal": metadata.goal,
                "timestamp": metadata.timestamp,
                "execution_path": metadata.execution_path,
                "verified_leaf_coverage": metadata.verified_leaf_coverage,
            }
            break

    # Build the raw CDG dict and sanitize it
    raw_cdg = {
        "nodes": serialized_nodes,
        "edges": serialized_edges,
    }
    return sanitize_cdg(raw_cdg)

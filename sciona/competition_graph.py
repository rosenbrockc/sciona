"""Restore version-bound competition contracts for review and implementation.

Intake snapshots contain proposed bindings, not executable approvals. Restored
nodes remain pending until a separate validated implementation version exists.
"""

import hashlib
import json
from typing import Any

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, NodeStatus
from sciona.architect.skeleton_assets import SkeletonFamilyAsset


def restore_competition_graph(snapshot: dict[str, Any], *, content_hash: str) -> CDGExport:
    encoded = json.dumps(snapshot, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if not content_hash or hashlib.sha256(encoded.encode()).hexdigest() != content_hash:
        raise ValueError("competition snapshot does not match requested version")
    if snapshot.get("format") != "competition-intake.v1":
        raise ValueError("unsupported competition snapshot format")
    raw = snapshot["template"]
    asset = SkeletonFamilyAsset.model_validate(raw)
    nodes = [AlgorithmicNode(
        node_id=stage.stage_id, name=stage.name, description=stage.description,
        concept_type=stage.concept_type, inputs=stage.inputs, outputs=stage.outputs,
        status=NodeStatus.PENDING, conceptual_summary=stage.dejargonized_description,
    ) for stage in asset.stages]
    edges = []
    for raw_edge in raw.get("edges", []):
        edge = dict(raw_edge)
        edge["source_id"] = edge.pop("source_stage_id")
        edge["target_id"] = edge.pop("target_stage_id")
        if edge.get("loss_class") == "lossy_but_allowed":
            edge["loss_class"] = "lossy_allowed"
        edges.append(DependencyEdge.model_validate(edge))
    return CDGExport(nodes=nodes, edges=edges, metadata={
        "artifact_kind": "cdg", "artifact_fqdn": "cdg.competition." + asset.asset_id,
        "artifact_content_hash": content_hash,
        "artifact_source": "competition_intake",
        "execution_ready": False,
        "boundary_inputs": raw.get("inputs", []),
        "boundary_outputs": raw.get("outputs", []),
        "planning_constraints": raw.get("planning_constraints", []),
        "applicability": raw.get("applicability", {}),
        "stage_contracts": raw["stages"],
        "edge_contracts": raw.get("edges", []),
        "proposed_bindings": snapshot["bindings"],
    })


def load_competition_graph(connection, *, version_id: str) -> CDGExport:
    """Load exactly one canonical snapshot; never combine latest audit rows."""
    rows = connection.execute(
        "SELECT v.content_hash,e.details FROM artifact_versions v "
        "JOIN artifact_audit_evidence e USING(version_id) "
        "WHERE v.version_id=%s AND e.runner_version='competition-intake.v1' "
        "AND e.audit_type='asset_integrity_check'",
        (version_id,),
    ).fetchall()
    if len(rows) != 1:
        raise ValueError("competition version requires exactly one intake snapshot")
    return restore_competition_graph(rows[0]["details"]["snapshot"], content_hash=rows[0]["content_hash"])

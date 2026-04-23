"""Conservative skeleton proposal generation for node enrichment.

Phase 2 only generates bounded, passive skeleton proposals. These candidates
are metadata objects and must not change live selection behavior yet.
"""

from __future__ import annotations

from sciona.architect.models import AlgorithmicNode, IOSpec, NodeStatus
from sciona.architect.proposal_models import EnrichmentProposal, proposal_placeholder_skeleton
from sciona.architect.skeletons import NAMED_SKELETONS, infer_boundary_ports

MAX_SKELETON_PROPOSAL_NODES = 6
MAX_SKELETON_PROPOSAL_EDGES = 8

# Keep the initial surface deliberately small. These are named skeletons with
# stable, moderate boundary structure and modest internal size.
ALLOWED_SKELETON_PROPOSALS: tuple[str, ...] = (
    "signal_detect_measure",
    "kalman_filter",
    "map_over",
    "fixed_point",
)


def _normalize_type_name(type_desc: str) -> str:
    return " ".join(str(type_desc or "").strip().lower().split())


def _port_types_compatible(expected: IOSpec, candidate: IOSpec) -> bool:
    """Conservative type compatibility: exact normalized match or explicit any."""
    lhs = _normalize_type_name(expected.type_desc)
    rhs = _normalize_type_name(candidate.type_desc)
    if not lhs or not rhs:
        return False
    if lhs == rhs:
        return True
    if lhs == "any":
        return True
    return False


def _boundary_compatible(node: AlgorithmicNode, inputs: list[IOSpec], outputs: list[IOSpec]) -> tuple[bool, float]:
    """Return whether a skeleton boundary is conservatively compatible."""
    if len(node.inputs) != len(inputs) or len(node.outputs) != len(outputs):
        return False, 0.0
    if not node.inputs and not node.outputs:
        return False, 0.0

    compatible_ports = 0
    total_ports = len(node.inputs) + len(node.outputs)
    for expected, candidate in zip(node.inputs, inputs):
        if not _port_types_compatible(expected, candidate):
            return False, 0.0
        compatible_ports += 1
    for expected, candidate in zip(node.outputs, outputs):
        if not _port_types_compatible(expected, candidate):
            return False, 0.0
        compatible_ports += 1
    return True, compatible_ports / max(total_ports, 1)


def generate_skeleton_proposals(
    node: AlgorithmicNode,
    *,
    allowlist: tuple[str, ...] = ALLOWED_SKELETON_PROPOSALS,
    max_nodes: int = MAX_SKELETON_PROPOSAL_NODES,
    max_edges: int = MAX_SKELETON_PROPOSAL_EDGES,
) -> list[EnrichmentProposal]:
    """Generate bounded skeleton proposals for an eligible node."""
    if node.status == NodeStatus.ATOMIC:
        return []

    proposals: list[EnrichmentProposal] = []
    for skeleton_name in allowlist:
        skeleton = NAMED_SKELETONS.get(skeleton_name)
        if skeleton is None:
            continue
        if len(skeleton.template_nodes) > max_nodes:
            continue
        if len(skeleton.template_edges) > max_edges:
            continue

        boundary_inputs, boundary_outputs = infer_boundary_ports(
            skeleton.template_nodes,
            skeleton.template_edges,
        )
        compatible, compatibility_score = _boundary_compatible(
            node,
            boundary_inputs,
            boundary_outputs,
        )
        if not compatible:
            continue

        concept_types = {
            child.concept_type.value for child in skeleton.template_nodes
        }
        proposals.append(
            proposal_placeholder_skeleton(
                skeleton_name=skeleton_name,
                source_family=skeleton.paradigm.value,
                source_label=skeleton.name,
                confidence=compatibility_score,
                compatibility_score=compatibility_score,
                delta_nodes=len(skeleton.template_nodes),
                delta_edges=len(skeleton.template_edges),
                delta_family_count=1,
                delta_concept_type_count=len(concept_types),
                payload={
                    "paradigm": skeleton.paradigm.value,
                    "boundary_input_names": [port.name for port in boundary_inputs],
                    "boundary_output_names": [port.name for port in boundary_outputs],
                    "source_asset": dict(
                        skeleton.metadata.get("asset", {})
                        or skeleton.metadata.get("source_asset", {})
                    ),
                },
            )
        )
    return proposals

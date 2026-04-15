"""CDG builders for the synthetic family fixture."""

from __future__ import annotations

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)


def synthetic_node(
    node_id: str,
    name: str,
    *,
    concept_type: ConceptType = ConceptType.CUSTOM,
    matched_primitive: str | None = None,
) -> AlgorithmicNode:
    """Build a minimal atomic node usable as a synthetic-family CDG vertex."""
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=[IOSpec(name="in", type_desc="synth_payload")],
        outputs=[IOSpec(name="out", type_desc="synth_payload")],
        type_signature=f"{name} -> synth_payload",
    )


def synthetic_edge(source_id: str, target_id: str) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name="out",
        input_name="in",
        source_type="synth_payload",
        target_type="synth_payload",
    )


def synthetic_cdg() -> CDGExport:
    """Return the canonical linear source→process→sink synthetic CDG."""
    nodes = [
        synthetic_node("source", "synth_source", matched_primitive="synth_source"),
        synthetic_node("process", "synth_process", matched_primitive="synth_process"),
        synthetic_node("sink", "synth_sink", matched_primitive="synth_sink"),
    ]
    edges = [
        synthetic_edge("source", "process"),
        synthetic_edge("process", "sink"),
    ]
    return CDGExport(nodes=nodes, edges=edges, metadata={})

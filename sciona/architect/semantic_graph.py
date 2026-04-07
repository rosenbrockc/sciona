"""Semantic CDG projection with first-class root-boundary representation.

This module adds an additive semantic layer on top of the existing CDGExport
form. It does not replace the executable graph; it projects root input/output
boundaries into explicit semantic ports and provides a focused lowering helper
for root-boundary interposition.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec


class SemanticBoundaryKind(str, Enum):
    """Kinds of first-class semantic boundary ports."""

    ROOT_INPUT = "root_input"
    ROOT_OUTPUT = "root_output"


class SemanticDataKind(str, Enum):
    """High-level information-flow kinds carried by semantic edges."""

    GENERIC = "generic"
    WAVEFORM = "waveform"
    EVENT_SEQUENCE = "event_sequence"
    RATE_SERIES = "rate_series"


class SemanticLossClass(str, Enum):
    """Coarse information-loss classification for a semantic edge."""

    UNKNOWN = "unknown"
    PRESERVING = "preserving"
    LOSSY = "lossy"


class SemanticEdgeProvenance(str, Enum):
    """How a semantic edge was derived."""

    DECLARED_EDGE = "declared_edge"
    ROOT_CONTRACT = "root_contract"


class SemanticBoundaryPort(BaseModel):
    """A first-class boundary port projected from a root node contract."""

    boundary_id: str
    root_node_id: str
    kind: SemanticBoundaryKind
    port: IOSpec


class SemanticFlowEdge(BaseModel):
    """An explicit semantic flow edge between nodes and/or boundaries."""

    source_id: str
    target_id: str
    output_name: str
    input_name: str
    data_kind: SemanticDataKind = SemanticDataKind.GENERIC
    loss_class: SemanticLossClass = SemanticLossClass.UNKNOWN
    provenance: SemanticEdgeProvenance = SemanticEdgeProvenance.DECLARED_EDGE


class SemanticBoundaryTarget(BaseModel):
    """Resolved node-side consumer of a semantic root boundary."""

    boundary_id: str
    node_id: str
    node_name: str
    port_name: str
    matched_primitive: str = ""
    concept_type: str = ""


class SemanticCDG(BaseModel):
    """Canonical semantic projection of an executable CDG."""

    nodes: list[AlgorithmicNode]
    boundaries: list[SemanticBoundaryPort]
    edges: list[SemanticFlowEdge]
    metadata: dict[str, Any] = Field(default_factory=dict)

    def boundary(self, boundary_id: str) -> SemanticBoundaryPort | None:
        """Return one semantic boundary by stable identifier."""
        for boundary in self.boundaries:
            if boundary.boundary_id == boundary_id:
                return boundary
        return None

    def find_root_input_consumers(
        self,
        input_name: str,
        *,
        matched_primitive: str = "",
    ) -> list[SemanticBoundaryTarget]:
        """Return unresolved consumers of a named root input boundary."""
        node_map = {node.node_id: node for node in self.nodes}
        matches: list[SemanticBoundaryTarget] = []
        for boundary in self.boundaries:
            if (
                boundary.kind != SemanticBoundaryKind.ROOT_INPUT
                or boundary.port.name != input_name
            ):
                continue
            for edge in self.edges:
                if edge.source_id != boundary.boundary_id:
                    continue
                node = node_map.get(edge.target_id)
                if node is None:
                    continue
                if matched_primitive and node.matched_primitive != matched_primitive:
                    continue
                matches.append(
                    SemanticBoundaryTarget(
                        boundary_id=boundary.boundary_id,
                        node_id=node.node_id,
                        node_name=node.name,
                        port_name=edge.input_name,
                        matched_primitive=str(node.matched_primitive or ""),
                        concept_type=node.concept_type.value,
                    )
                )
        return matches


def _boundary_id(root_node_id: str, kind: SemanticBoundaryKind, port_name: str) -> str:
    prefix = "in" if kind == SemanticBoundaryKind.ROOT_INPUT else "out"
    return f"boundary:{prefix}:{root_node_id}:{port_name}"


def _infer_data_kind(*tokens: str) -> SemanticDataKind:
    joined = " ".join(tokens).lower()
    if any(token in joined for token in ("signal", "waveform", "ecg", "raw trace")):
        return SemanticDataKind.WAVEFORM
    if any(token in joined for token in ("event", "peak", "beat", "r_peak")):
        return SemanticDataKind.EVENT_SEQUENCE
    if any(token in joined for token in ("rate", "hr", "bpm", "frequency")):
        return SemanticDataKind.RATE_SERIES
    return SemanticDataKind.GENERIC


def _incoming_index(cdg: CDGExport) -> set[tuple[str, str]]:
    return {(edge.target_id, edge.input_name) for edge in cdg.edges}


def _outgoing_index(cdg: CDGExport) -> set[tuple[str, str]]:
    return {(edge.source_id, edge.output_name) for edge in cdg.edges}


def project_semantic_cdg(cdg: CDGExport) -> SemanticCDG:
    """Project a CDGExport into a semantic graph with explicit boundaries."""
    node_map = {node.node_id: node for node in cdg.nodes}
    incoming = _incoming_index(cdg)
    outgoing = _outgoing_index(cdg)

    semantic_edges: list[SemanticFlowEdge] = [
        SemanticFlowEdge(
            source_id=edge.source_id,
            target_id=edge.target_id,
            output_name=edge.output_name,
            input_name=edge.input_name,
            data_kind=_infer_data_kind(
                edge.output_name,
                edge.input_name,
                edge.source_type,
                edge.target_type,
            ),
            loss_class=SemanticLossClass.PRESERVING,
            provenance=SemanticEdgeProvenance.DECLARED_EDGE,
        )
        for edge in cdg.edges
    ]
    boundaries: list[SemanticBoundaryPort] = []

    roots = [node for node in cdg.nodes if node.parent_id is None]
    for root in roots:
        scope_ids = set(root.children) if root.children else {
            node.node_id for node in cdg.nodes if node.node_id != root.node_id
        }

        for port in root.inputs:
            boundary = SemanticBoundaryPort(
                boundary_id=_boundary_id(
                    root.node_id,
                    SemanticBoundaryKind.ROOT_INPUT,
                    port.name,
                ),
                root_node_id=root.node_id,
                kind=SemanticBoundaryKind.ROOT_INPUT,
                port=port.model_copy(),
            )
            boundaries.append(boundary)
            for node_id in scope_ids:
                node = node_map.get(node_id)
                if node is None:
                    continue
                for node_input in node.inputs:
                    if node_input.name != port.name:
                        continue
                    if (node.node_id, node_input.name) in incoming:
                        continue
                    semantic_edges.append(
                        SemanticFlowEdge(
                            source_id=boundary.boundary_id,
                            target_id=node.node_id,
                            output_name=port.name,
                            input_name=node_input.name,
                            data_kind=_infer_data_kind(
                                port.name,
                                node_input.name,
                                port.type_desc,
                                node_input.type_desc,
                            ),
                            loss_class=SemanticLossClass.PRESERVING,
                            provenance=SemanticEdgeProvenance.ROOT_CONTRACT,
                        )
                    )

        for port in root.outputs:
            boundary = SemanticBoundaryPort(
                boundary_id=_boundary_id(
                    root.node_id,
                    SemanticBoundaryKind.ROOT_OUTPUT,
                    port.name,
                ),
                root_node_id=root.node_id,
                kind=SemanticBoundaryKind.ROOT_OUTPUT,
                port=port.model_copy(),
            )
            boundaries.append(boundary)
            for node_id in scope_ids:
                node = node_map.get(node_id)
                if node is None:
                    continue
                for node_output in node.outputs:
                    if node_output.name != port.name:
                        continue
                    if (node.node_id, node_output.name) in outgoing:
                        continue
                    semantic_edges.append(
                        SemanticFlowEdge(
                            source_id=node.node_id,
                            target_id=boundary.boundary_id,
                            output_name=node_output.name,
                            input_name=port.name,
                            data_kind=_infer_data_kind(
                                node_output.name,
                                port.name,
                                node_output.type_desc,
                                port.type_desc,
                            ),
                            loss_class=SemanticLossClass.PRESERVING,
                            provenance=SemanticEdgeProvenance.ROOT_CONTRACT,
                        )
                    )

    return SemanticCDG(
        nodes=[node.model_copy(deep=True) for node in cdg.nodes],
        boundaries=boundaries,
        edges=semantic_edges,
        metadata=dict(cdg.metadata or {}),
    )


def insert_node_before_root_input_consumer(
    cdg: CDGExport,
    *,
    root_input_name: str,
    target_primitive: str,
    inserted_node: AlgorithmicNode,
    target_input_name: str | None = None,
) -> CDGExport:
    """Lower a root-boundary interposition into the executable CDG form.

    This helper finds the first unresolved consumer of a declared root input
    and inserts a new executable node immediately before that consumer. The
    inserted node remains root-fed by leaving its own input unresolved, which
    matches the current assembler/runtime convention.
    """
    semantic = project_semantic_cdg(cdg)
    node_map = {node.node_id: node for node in cdg.nodes}
    candidates: list[tuple[SemanticBoundaryPort, SemanticFlowEdge, AlgorithmicNode]] = []

    for boundary in semantic.boundaries:
        if (
            boundary.kind != SemanticBoundaryKind.ROOT_INPUT
            or boundary.port.name != root_input_name
        ):
            continue
        for edge in semantic.edges:
            if edge.source_id != boundary.boundary_id:
                continue
            if target_input_name is not None and edge.input_name != target_input_name:
                continue
            candidate = node_map.get(edge.target_id)
            if candidate is None:
                continue
            if candidate.matched_primitive != target_primitive:
                continue
            candidates.append((boundary, edge, candidate))

    if not candidates:
        raise ValueError(
            f"No unresolved root-input consumer found for '{root_input_name}' -> "
            f"'{target_primitive}'."
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous root-input consumer for '{root_input_name}' -> "
            f"'{target_primitive}': {len(candidates)} matches."
        )

    boundary, edge, target_node = candidates[0]
    root_node = node_map.get(boundary.root_node_id)
    if root_node is None:
        raise ValueError(f"Root node '{boundary.root_node_id}' not found in CDG.")

    new_node_id = inserted_node.node_id
    existing_ids = {node.node_id for node in cdg.nodes}
    if not new_node_id or new_node_id in existing_ids:
        new_node_id = f"semantic_{uuid.uuid4().hex[:8]}"

    new_node = inserted_node.model_copy(
        deep=True,
        update={
            "node_id": new_node_id,
            "parent_id": inserted_node.parent_id or root_node.node_id,
            "depth": inserted_node.depth or (root_node.depth + 1),
        },
    )

    output_port = new_node.outputs[0] if new_node.outputs else IOSpec(
        name=edge.output_name,
        type_desc=boundary.port.type_desc,
    )
    target_input = next(
        (
            port
            for port in target_node.inputs
            if port.name == edge.input_name
        ),
        IOSpec(name=edge.input_name, type_desc=edge.input_name),
    )

    rewritten_root = root_node.model_copy(
        update={
            "children": [
                *root_node.children,
                *([] if new_node.node_id in root_node.children else [new_node.node_id]),
            ]
        }
    )

    new_nodes: list[AlgorithmicNode] = []
    for node in cdg.nodes:
        if node.node_id == root_node.node_id:
            new_nodes.append(rewritten_root)
        else:
            new_nodes.append(node.model_copy(deep=True))
    new_nodes.append(new_node)

    new_edges = [edge.model_copy(deep=True) for edge in cdg.edges]
    new_edges.append(
        DependencyEdge(
            source_id=new_node.node_id,
            target_id=target_node.node_id,
            output_name=output_port.name,
            input_name=edge.input_name,
            source_type=output_port.type_desc,
            target_type=target_input.type_desc,
        )
    )

    return cdg.model_copy(
        update={
            "nodes": new_nodes,
            "edges": new_edges,
        }
    )

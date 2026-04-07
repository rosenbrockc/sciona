"""Boundary-aware rewrite helpers lowered onto executable CDGs."""

from __future__ import annotations

import uuid
from typing import Callable

from sciona.architect.graph_rewriter import GraphState
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode
from sciona.architect.semantic_graph import (
    insert_node_before_root_input_consumer,
    project_semantic_cdg,
)


def build_boundary_interposition_callback(
    *,
    target_primitive: str,
    boundary_input_name: str,
    insert_node: AlgorithmicNode,
    target_input_name: str = "",
    insert_output_name: str = "",
) -> Callable[[CDGExport], GraphState[CDGExport]]:
    """Create a semantic fallback that interposes a node at a root boundary."""

    def _apply(graph: CDGExport) -> GraphState[CDGExport]:
        target_input = target_input_name or boundary_input_name
        original_node_ids = {node.node_id for node in graph.nodes}
        semantic = project_semantic_cdg(graph)
        matches = semantic.find_root_input_consumers(
            boundary_input_name,
            matched_primitive=target_primitive,
        )
        matches = [match for match in matches if match.port_name == target_input]
        if not matches:
            return GraphState.failure(
                f"no root-boundary consumer for '{boundary_input_name}' into '{target_primitive}'"
            )
        if len(matches) > 1:
            return GraphState.failure(
                f"ambiguous root-boundary consumer for '{boundary_input_name}' into '{target_primitive}'"
            )

        try:
            rewritten = insert_node_before_root_input_consumer(
                graph,
                root_input_name=boundary_input_name,
                target_primitive=target_primitive,
                inserted_node=insert_node.model_copy(
                    deep=True,
                    update={"node_id": f"{insert_node.node_id}_{uuid.uuid4().hex[:8]}"},
                ),
                target_input_name=target_input,
            )
        except ValueError as exc:
            return GraphState.failure(str(exc))

        root_node = next((node for node in graph.nodes if node.parent_id is None), None)
        inserted = next(
            (
                node
                for node in rewritten.nodes
                if node.node_id not in original_node_ids
                and node.matched_primitive == insert_node.matched_primitive
            ),
            None,
        )
        if root_node is not None and inserted is not None:
            normalized_nodes: list[AlgorithmicNode] = []
            for node in rewritten.nodes:
                if node.node_id == inserted.node_id:
                    normalized_nodes.append(
                        node.model_copy(update={"parent_id": root_node.node_id})
                    )
                    continue
                if node.node_id == root_node.node_id:
                    children = list(node.children)
                    if inserted.node_id not in children:
                        children.append(inserted.node_id)
                    normalized_nodes.append(node.model_copy(update={"children": children}))
                    continue
                normalized_nodes.append(node)
            rewritten = rewritten.model_copy(update={"nodes": normalized_nodes})

        metadata = dict(rewritten.metadata or {})
        metadata["semantic_rewrite"] = {
            "kind": "boundary_interposition",
            "target_primitive": target_primitive,
            "boundary_input_name": boundary_input_name,
            "target_input_name": target_input,
            "insert_node_primitive": insert_node.matched_primitive,
            "insert_output_name": insert_output_name or target_input,
        }
        return GraphState.success(rewritten.model_copy(update={"metadata": metadata}))

    return _apply

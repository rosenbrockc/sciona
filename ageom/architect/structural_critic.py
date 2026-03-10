"""Deterministic structural validation for architect decompositions."""

from __future__ import annotations

from collections import deque
from typing import Any

from ageom.architect.models import AlgorithmicNode, DependencyEdge, NodeStatus


def _is_any_type(type_desc: str) -> bool:
    return type_desc.strip() in {"", "Any"}


def _tokenize(text: str) -> set[str]:
    normalized = text.lower().replace("_", " ").replace("-", " ")
    return {tok for tok in normalized.split() if tok}


def _check_basic_structure(
    parent: AlgorithmicNode,
    children: list[AlgorithmicNode],
    child_edges: list[DependencyEdge],
    *,
    max_depth: int,
    catalog: Any,
) -> list[str]:
    issues: list[str] = []

    if len(children) < 2:
        issues.append(f"Need at least 2 sub-nodes, got {len(children)}")

    for child in children:
        if child.depth > max_depth:
            issues.append(
                f"Node '{child.name}' exceeds max depth ({child.depth} > {max_depth})"
            )

    for edge in child_edges:
        if edge.source_id == edge.target_id:
            issues.append(f"Self-loop detected on edge {edge.source_id}")

    child_by_id = {c.node_id: c for c in children}
    for edge in child_edges:
        src = child_by_id.get(edge.source_id)
        tgt = child_by_id.get(edge.target_id)
        if src and src.outputs:
            out_names = {o.name for o in src.outputs}
            if edge.output_name not in out_names and out_names:
                issues.append(
                    f"Edge output '{edge.output_name}' not in "
                    f"source '{src.name}' outputs: {out_names}"
                )
        if tgt and tgt.inputs:
            in_names = {i.name for i in tgt.inputs}
            if edge.input_name not in in_names and in_names:
                issues.append(
                    f"Edge input '{edge.input_name}' not in "
                    f"target '{tgt.name}' inputs: {in_names}"
                )

    for child in children:
        if child.status == NodeStatus.ATOMIC and not catalog.is_atomic(child):
            issues.append(f"Node '{child.name}' claims atomic but not in catalog")
        if (
            child.status == NodeStatus.ATOMIC
            and child.matched_primitive
            and child.primitive_binding_source == "token_overlap"
            and child.primitive_binding_confidence < 0.75
        ):
            issues.append(
                f"Node '{child.name}' has weak primitive binding "
                f"({child.matched_primitive}, confidence={child.primitive_binding_confidence:.2f})"
            )

    parent_is_typed = any(
        not _is_any_type(io.type_desc) for io in parent.inputs + parent.outputs
    )
    if parent_is_typed:
        for child in children:
            weak_inputs = [io.name for io in child.inputs if _is_any_type(io.type_desc)]
            weak_outputs = [io.name for io in child.outputs if _is_any_type(io.type_desc)]
            if weak_inputs or weak_outputs:
                issues.append(
                    f"Node '{child.name}' uses unresolved Any ports "
                    f"(inputs={weak_inputs}, outputs={weak_outputs})"
                )
        for edge in child_edges:
            if _is_any_type(edge.source_type) or _is_any_type(edge.target_type):
                issues.append(
                    "Edge "
                    f"{edge.source_id}->{edge.target_id} uses unresolved Any types "
                    f"({edge.source_type} -> {edge.target_type})"
                )
    return issues


def _check_io_coverage(
    parent: AlgorithmicNode,
    children: list[AlgorithmicNode],
) -> list[str]:
    issues: list[str] = []
    parent_input_names = {io.name for io in parent.inputs}
    parent_output_names = {io.name for io in parent.outputs}
    child_input_names = {io.name for child in children for io in child.inputs}
    child_output_names = {io.name for child in children for io in child.outputs}

    uncovered_inputs = sorted(parent_input_names - child_input_names)
    if uncovered_inputs:
        issues.append(f"Parent inputs not consumed: {uncovered_inputs}")

    uncovered_outputs = sorted(parent_output_names - child_output_names)
    if uncovered_outputs:
        issues.append(f"Parent outputs not produced: {uncovered_outputs}")

    return issues


def _check_duplicate_children(
    children: list[AlgorithmicNode],
    *,
    threshold: float = 0.85,
) -> list[str]:
    issues: list[str] = []
    for idx, left in enumerate(children):
        left_tokens = _tokenize(left.name)
        if not left_tokens:
            continue
        for right in children[idx + 1 :]:
            right_tokens = _tokenize(right.name)
            if not right_tokens:
                continue
            overlap = len(left_tokens & right_tokens)
            union = len(left_tokens | right_tokens)
            similarity = overlap / union if union else 0.0
            if similarity >= threshold:
                issues.append(
                    f"Near-duplicate child nodes: '{left.name}' and '{right.name}'"
                )
    return issues


def _check_output_reachability(
    parent: AlgorithmicNode,
    children: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> list[str]:
    if not parent.outputs or not edges:
        return []

    child_by_id = {child.node_id: child for child in children}
    consumers: dict[str, list[DependencyEdge]] = {}
    for edge in edges:
        consumers.setdefault(edge.source_id, []).append(edge)

    parent_input_names = {io.name for io in parent.inputs}
    reachable: set[str] = set()
    frontier: deque[str] = deque()

    for child in children:
        child_inputs = {io.name for io in child.inputs}
        if not child_inputs or child_inputs & parent_input_names:
            reachable.add(child.node_id)
            frontier.append(child.node_id)

    while frontier:
        source_id = frontier.popleft()
        source = child_by_id.get(source_id)
        source_outputs = {io.name for io in source.outputs} if source else set()
        for edge in consumers.get(source_id, []):
            if source_outputs and edge.output_name not in source_outputs:
                continue
            target = child_by_id.get(edge.target_id)
            target_inputs = {io.name for io in target.inputs} if target else set()
            if target_inputs and edge.input_name not in target_inputs:
                continue
            if edge.target_id not in reachable:
                reachable.add(edge.target_id)
                frontier.append(edge.target_id)

    issues: list[str] = []
    for output in parent.outputs:
        producers = [
            child.node_id for child in children if output.name in {io.name for io in child.outputs}
        ]
        if producers and not any(producer in reachable for producer in producers):
            issues.append(
                f"Parent output '{output.name}' is not reachable from parent inputs through child edges"
            )
    return issues


def structural_critique_issues(
    parent: AlgorithmicNode,
    children: list[AlgorithmicNode],
    child_edges: list[DependencyEdge],
    *,
    max_depth: int,
    catalog: Any,
) -> list[str]:
    """Return deterministic structural critique issues for a decomposition."""
    issues: list[str] = []
    issues.extend(
        _check_basic_structure(
            parent,
            children,
            child_edges,
            max_depth=max_depth,
            catalog=catalog,
        )
    )
    issues.extend(_check_io_coverage(parent, children))
    issues.extend(_check_duplicate_children(children))
    issues.extend(_check_output_reachability(parent, children, child_edges))
    return issues

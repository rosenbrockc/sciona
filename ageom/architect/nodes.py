"""Node functions and routing for the LangGraph decomposition cycle.

Each node function: async def fn(state, config) -> dict
Routing functions: def fn(state) -> str
"""

from __future__ import annotations

import json
import uuid
from typing import Any

from langchain_core.runnables import RunnableConfig

from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from ageom.architect.prompts import (
    CRITIQUE_SYSTEM,
    CRITIQUE_USER,
    DECOMPOSE_NODE_SYSTEM,
    DECOMPOSE_NODE_USER,
    SELECT_STRATEGY_SYSTEM,
    SELECT_STRATEGY_USER,
)
from ageom.architect.skeletons import SKELETON_TEMPLATES, instantiate_skeleton
from ageom.architect.state import DecompositionDeps, DecompositionState


def _get_deps(config: RunnableConfig) -> DecompositionDeps:
    """Extract DecompositionDeps from LangGraph config."""
    return config["configurable"]["deps"]


def _find_node(nodes: list[AlgorithmicNode], node_id: str) -> AlgorithmicNode | None:
    """Find a node by ID in the nodes list."""
    for n in nodes:
        if n.node_id == node_id:
            return n
    return None


def _format_io(specs: list[IOSpec]) -> str:
    """Format IOSpec list for prompt display."""
    if not specs:
        return "none"
    return ", ".join(f"{s.name}: {s.type_desc}" for s in specs)


def _format_primitives(prims: list) -> str:
    """Format primitives list for prompt display."""
    if not prims:
        return "No relevant primitives found."
    lines = []
    for p in prims[:10]:
        line = f"- {p.name} [{p.category.value}]: {p.description[:100]}"
        if p.type_signature:
            line += f"  (type: {p.type_signature[:60]})"
        lines.append(line)
    return "\n".join(lines)


def _parse_json(text: str) -> dict | None:
    """Try to parse JSON from LLM output, stripping markdown fences."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Drop first and last lines (fences)
        lines = [l for l in lines[1:] if not l.strip().startswith("```")]
        text = "\n".join(lines)
    try:
        return json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Node: select_strategy
# ---------------------------------------------------------------------------

async def select_strategy(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Entry point: LLM picks a paradigm and bootstraps the CDG via skeleton."""
    deps = _get_deps(config)
    goal = state["goal"]
    max_depth = state["max_depth"]

    available = list(SKELETON_TEMPLATES.keys())
    available_str = "\n".join(f"  - {ct.value}" for ct in available)

    response = await deps.llm.complete(
        SELECT_STRATEGY_SYSTEM.format(available_paradigms=available_str),
        SELECT_STRATEGY_USER.format(goal=goal),
    )

    parsed = _parse_json(response)

    # Parse paradigm or fall back to CUSTOM
    paradigm = ConceptType.CUSTOM
    variant_hint = ""
    if parsed:
        paradigm_str = parsed.get("paradigm", "")
        for ct in ConceptType:
            if ct.value == paradigm_str:
                paradigm = ct
                break
        variant_hint = parsed.get("variant_hint", "")

    # Create root node
    root_id = f"root_{uuid.uuid4().hex[:8]}"
    root = AlgorithmicNode(
        node_id=root_id,
        name=goal,
        description=goal,
        concept_type=paradigm,
        status=NodeStatus.DECOMPOSED,
        depth=0,
    )

    nodes: list[AlgorithmicNode] = [root]
    edges: list[DependencyEdge] = []
    skeleton_instantiated = False

    # Try to instantiate skeleton
    skeleton = SKELETON_TEMPLATES.get(paradigm)
    if skeleton:
        skel_nodes, skel_edges = instantiate_skeleton(
            skeleton, goal, parent_id=root_id, base_depth=0
        )
        root = root.model_copy(
            update={"children": [n.node_id for n in skel_nodes]}
        )
        nodes = [root] + skel_nodes
        edges = skel_edges
        skeleton_instantiated = True

    # Check which skeleton nodes are already atomic
    for i, node in enumerate(nodes):
        if node.node_id == root_id:
            continue
        if deps.catalog.is_atomic(node):
            # Find the matching primitive name
            prim_name = node.matched_primitive
            if not prim_name:
                name_lower = node.name.lower().replace(" ", "_")
                for p in deps.catalog.all_primitives():
                    if p.name.lower().replace(" ", "_") == name_lower:
                        prim_name = p.name
                        break
            nodes[i] = node.model_copy(
                update={
                    "status": NodeStatus.ATOMIC,
                    "matched_primitive": prim_name or node.name,
                }
            )

    # Build pending queue (non-root, non-atomic)
    pending = [
        n.node_id for n in nodes
        if n.status == NodeStatus.PENDING
    ]

    current_node_id = pending[0] if pending else ""

    history_entry = {
        "step": "select_strategy",
        "paradigm": paradigm.value,
        "variant_hint": variant_hint,
        "skeleton_instantiated": skeleton_instantiated,
        "num_nodes": len(nodes),
        "num_pending": len(pending),
    }

    return {
        "nodes": nodes,
        "edges": edges,
        "history": [history_entry],
        "pending_node_ids": pending,
        "current_node_id": current_node_id,
        "paradigm": paradigm.value,
        "skeleton_instantiated": skeleton_instantiated,
        "critique_passed": False,
        "critique_reason": "",
        "critique_retries": 0,
        "done": len(pending) == 0,
        "error": "",
    }


# ---------------------------------------------------------------------------
# Node: decompose_node
# ---------------------------------------------------------------------------

async def decompose_node(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Core LLM decomposition: break current_node_id into sub-nodes + edges."""
    deps = _get_deps(config)
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    max_depth = state["max_depth"]

    node = _find_node(all_nodes, current_id)
    if node is None:
        return {
            "error": f"Node {current_id} not found",
            "done": True,
            "history": [{"step": "decompose_node", "error": f"Node {current_id} not found"}],
        }

    # Gather relevant primitives
    catalog_prims = deps.catalog.find_matching_primitives(node, k=5)
    try:
        skill_prims = deps.skill_index.search(
            f"{node.name} {node.description}", k=5
        )
    except Exception:
        skill_prims = []

    # Deduplicate by name
    seen_names: set[str] = set()
    all_prims = []
    for p in catalog_prims + skill_prims:
        if p.name not in seen_names:
            all_prims.append(p)
            seen_names.add(p.name)

    # Build retry context
    retry_context = ""
    retries = state.get("critique_retries", 0)
    if retries > 0:
        reason = state.get("critique_reason", "")
        retry_context = (
            f"IMPORTANT: This is retry #{retries}. "
            f"Previous decomposition was rejected: {reason}\n"
            "Please fix the issues and try again."
        )

    response = await deps.llm.complete(
        DECOMPOSE_NODE_SYSTEM,
        DECOMPOSE_NODE_USER.format(
            node_name=node.name,
            node_description=node.description,
            concept_type=node.concept_type.value,
            inputs=_format_io(node.inputs),
            outputs=_format_io(node.outputs),
            depth=node.depth,
            max_depth=max_depth,
            primitives=_format_primitives(all_prims),
            retry_context=retry_context,
        ),
    )

    parsed = _parse_json(response)
    if not parsed:
        # Fallback: empty decomposition → critic rejects → retry
        return {
            "nodes": [],
            "edges": [],
            "history": [{"step": "decompose_node", "node_id": current_id, "parse_error": True}],
        }

    # Parse sub-nodes
    new_nodes: list[AlgorithmicNode] = []
    name_to_id: dict[str, str] = {}

    for sub in parsed.get("sub_nodes", []):
        sub_id = f"node_{uuid.uuid4().hex[:8]}"
        sub_name = sub.get("name", "unnamed")
        name_to_id[sub_name] = sub_id

        # Parse concept_type
        concept_type = node.concept_type
        ct_str = sub.get("concept_type", "")
        for ct in ConceptType:
            if ct.value == ct_str:
                concept_type = ct
                break

        # Parse I/O
        inputs = [
            IOSpec(name=io.get("name", ""), type_desc=io.get("type_desc", "any"))
            for io in sub.get("inputs", [])
        ]
        outputs = [
            IOSpec(name=io.get("name", ""), type_desc=io.get("type_desc", "any"))
            for io in sub.get("outputs", [])
        ]

        is_atomic = sub.get("is_atomic", False)
        matched_prim = sub.get("matched_primitive")

        sub_node = AlgorithmicNode(
            node_id=sub_id,
            parent_id=current_id,
            name=sub_name,
            description=sub.get("description", ""),
            concept_type=concept_type,
            inputs=inputs,
            outputs=outputs,
            depth=node.depth + 1,
            type_signature=sub.get("type_signature", ""),
            matched_primitive=matched_prim if matched_prim != "null" else None,
            status=NodeStatus.PENDING,
        )

        # Validate atomic claims against catalog
        if is_atomic and deps.catalog.is_atomic(sub_node):
            sub_node = sub_node.model_copy(update={"status": NodeStatus.ATOMIC})
        elif is_atomic:
            # LLM claimed atomic but catalog doesn't confirm — keep as PENDING
            sub_node = sub_node.model_copy(update={"matched_primitive": None})

        new_nodes.append(sub_node)

    # Parse edges
    new_edges: list[DependencyEdge] = []
    for edge_spec in parsed.get("edges", []):
        src_name = edge_spec.get("source_name", "")
        tgt_name = edge_spec.get("target_name", "")
        src_id = name_to_id.get(src_name, "")
        tgt_id = name_to_id.get(tgt_name, "")
        if src_id and tgt_id:
            data_type = edge_spec.get("data_type", "any")
            new_edges.append(DependencyEdge(
                source_id=src_id,
                target_id=tgt_id,
                output_name=edge_spec.get("output_name", "result"),
                input_name=edge_spec.get("input_name", "data"),
                source_type=data_type,
                target_type=data_type,
            ))

    history_entry = {
        "step": "decompose_node",
        "node_id": current_id,
        "num_sub_nodes": len(new_nodes),
        "num_edges": len(new_edges),
    }

    return {
        "nodes": new_nodes,
        "edges": new_edges,
        "history": [history_entry],
    }


# ---------------------------------------------------------------------------
# Node: critique_decomposition
# ---------------------------------------------------------------------------

async def critique_decomposition(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """Two-phase validation: deterministic checks first, then LLM critique."""
    deps = _get_deps(config)
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    max_depth = state["max_depth"]

    parent = _find_node(all_nodes, current_id)
    if parent is None:
        return {
            "critique_passed": False,
            "critique_reason": f"Parent node {current_id} not found",
            "history": [{"step": "critique", "error": "parent not found"}],
        }

    # Find children of current node
    children = [n for n in all_nodes if n.parent_id == current_id and n.status != NodeStatus.REJECTED]
    child_edges = [
        e for e in state["edges"]
        if e.source_id in {c.node_id for c in children}
        or e.target_id in {c.node_id for c in children}
    ]

    # ------------------------------------------------------------------
    # Phase A: Deterministic checks (fast, free)
    # ------------------------------------------------------------------
    issues: list[str] = []

    # Check: at least 2 children
    if len(children) < 2:
        issues.append(f"Need at least 2 sub-nodes, got {len(children)}")

    # Check: depth constraint
    for child in children:
        if child.depth > max_depth:
            issues.append(f"Node '{child.name}' exceeds max depth ({child.depth} > {max_depth})")

    # Check: no self-loops in edges
    for edge in child_edges:
        if edge.source_id == edge.target_id:
            issues.append(f"Self-loop detected on edge {edge.source_id}")

    # Check: edge I/O name validity
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

    # Check: atomic claims match catalog
    for child in children:
        if child.status == NodeStatus.ATOMIC and not deps.catalog.is_atomic(child):
            issues.append(f"Node '{child.name}' claims atomic but not in catalog")

    if issues:
        reason = "Deterministic checks failed: " + "; ".join(issues)
        return {
            "critique_passed": False,
            "critique_reason": reason,
            "history": [{"step": "critique", "phase": "deterministic", "issues": issues}],
        }

    # ------------------------------------------------------------------
    # Phase B: LLM critique (only if Phase A passes)
    # ------------------------------------------------------------------
    sub_nodes_str = "\n".join(
        f"  - {c.name} [{c.concept_type.value}] "
        f"(inputs: {_format_io(c.inputs)}, outputs: {_format_io(c.outputs)}, "
        f"status: {c.status.value})"
        for c in children
    )
    edges_str = "\n".join(
        f"  - {e.source_id[:12]}... -> {e.target_id[:12]}...: "
        f"{e.output_name} -> {e.input_name} ({e.source_type})"
        for e in child_edges
    )

    catalog_prims = deps.catalog.find_matching_primitives(parent, k=5)

    response = await deps.llm.complete(
        CRITIQUE_SYSTEM,
        CRITIQUE_USER.format(
            parent_name=parent.name,
            parent_description=parent.description,
            parent_inputs=_format_io(parent.inputs),
            parent_outputs=_format_io(parent.outputs),
            sub_nodes=sub_nodes_str,
            edges=edges_str or "  (no edges)",
            current_depth=parent.depth,
            max_depth=max_depth,
            primitives=_format_primitives(catalog_prims),
        ),
    )

    parsed = _parse_json(response)

    if parsed is None:
        # Conservative fallback: reject
        return {
            "critique_passed": False,
            "critique_reason": "Failed to parse LLM critique response",
            "history": [{"step": "critique", "phase": "llm", "parse_error": True}],
        }

    approved = parsed.get("approved", False)
    reason = parsed.get("reason", "")
    flagged = parsed.get("flagged_nodes", [])

    # Mark flagged nodes as HIGH_RISK via state update
    flagged_updates: list[AlgorithmicNode] = []
    if flagged:
        for child in children:
            if child.name in flagged:
                flagged_updates.append(
                    child.model_copy(update={"status": NodeStatus.HIGH_RISK})
                )

    result: dict[str, Any] = {
        "critique_passed": approved,
        "critique_reason": reason,
        "history": [{"step": "critique", "phase": "llm", "approved": approved, "reason": reason}],
    }
    if flagged_updates:
        result["nodes"] = flagged_updates

    return result


# ---------------------------------------------------------------------------
# Node: advance_node
# ---------------------------------------------------------------------------

async def advance_node(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """After approved critique or max-retries exhaustion: move to next node."""
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    pending = list(state["pending_node_ids"])
    retries = state.get("critique_retries", 0)
    critique_passed = state.get("critique_passed", False)
    deps = _get_deps(config)

    parent = _find_node(all_nodes, current_id)
    updated_nodes: list[AlgorithmicNode] = []

    if critique_passed and parent:
        # Update parent: mark as DECOMPOSED with children
        children = [
            n for n in all_nodes
            if n.parent_id == current_id and n.status != NodeStatus.REJECTED
        ]
        child_ids = [c.node_id for c in children]
        updated_nodes.append(
            parent.model_copy(update={
                "status": NodeStatus.DECOMPOSED,
                "children": child_ids,
            })
        )

        # Add non-atomic children to pending
        new_pending = [
            c.node_id for c in children
            if c.status == NodeStatus.PENDING
        ]
    else:
        # Max retries exhausted: mark node as HIGH_RISK, move on
        if parent:
            updated_nodes.append(
                parent.model_copy(update={"status": NodeStatus.HIGH_RISK})
            )
        new_pending = []

    # Remove current from pending
    if current_id in pending:
        pending.remove(current_id)

    # Add new pending nodes
    pending.extend(new_pending)

    # Pick next
    next_id = pending[0] if pending else ""
    done = len(pending) == 0

    return {
        "nodes": updated_nodes,
        "pending_node_ids": pending,
        "current_node_id": next_id,
        "critique_retries": 0,
        "critique_passed": False,
        "critique_reason": "",
        "done": done,
        "history": [{
            "step": "advance_node",
            "from_node": current_id,
            "next_node": next_id,
            "done": done,
        }],
    }


# ---------------------------------------------------------------------------
# Node: prepare_retry
# ---------------------------------------------------------------------------

async def prepare_retry(
    state: DecompositionState, config: RunnableConfig
) -> dict[str, Any]:
    """After rejected critique: increment retries, mark children REJECTED, loop back."""
    current_id = state["current_node_id"]
    all_nodes = state["nodes"]
    retries = state.get("critique_retries", 0)

    # Mark rejected children
    rejected_updates: list[AlgorithmicNode] = []
    for node in all_nodes:
        if node.parent_id == current_id and node.status not in (
            NodeStatus.REJECTED, NodeStatus.ATOMIC
        ):
            rejected_updates.append(
                node.model_copy(update={"status": NodeStatus.REJECTED})
            )

    return {
        "nodes": rejected_updates,
        "critique_retries": retries + 1,
        "history": [{
            "step": "prepare_retry",
            "node_id": current_id,
            "retry_num": retries + 1,
            "num_rejected": len(rejected_updates),
        }],
    }


# ---------------------------------------------------------------------------
# Routing functions
# ---------------------------------------------------------------------------

def route_after_critic(state: DecompositionState) -> str:
    """Conditional edge after critique_decomposition."""
    passed = state.get("critique_passed", False)
    retries = state.get("critique_retries", 0)

    if not passed and retries < 3:
        return "retry_decompose"
    # Both "passed" and "max retries exhausted" go to advance_node
    return "next_node"


def route_after_advance(state: DecompositionState) -> str:
    """Conditional edge after advance_node."""
    if state.get("done", False):
        return "end"
    pending = state.get("pending_node_ids", [])
    if not pending:
        return "end"
    return "decompose"

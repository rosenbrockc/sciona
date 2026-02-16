"""CDG serialization and handoff to Round 2 (the Hunter agent).

Converts a completed Conceptual Dependency Graph into PDGNode objects
that the Round 2 matcher can ground into verified library functions.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from pydantic import BaseModel, Field

from ageom.architect.models import (
    AlgorithmicNode,
    DependencyEdge,
    NodeStatus,
)
from ageom.types import PDGNode, Prover


class HandoffValidationError(ValueError):
    """Raised when a CDG fails strict handoff validation."""

    def __init__(self, issues: list[str]) -> None:
        self.issues = issues
        super().__init__(
            f"Handoff validation failed with {len(issues)} issue(s): "
            + "; ".join(issues)
        )


class CDGExport(BaseModel):
    """A complete CDG ready for Round 2 handoff."""

    nodes: list[AlgorithmicNode]
    edges: list[DependencyEdge]
    metadata: dict = Field(default_factory=dict)

    def leaf_nodes(self) -> list[AlgorithmicNode]:
        """Return all leaf (atomic) nodes in the CDG."""
        return [n for n in self.nodes if n.status == NodeStatus.ATOMIC]

    def handoff_issues(self) -> list[str]:
        """Convenience: return validation issues for Round 2 handoff."""
        return validate_handoff(self)

    def non_atomic_leaves(self) -> list[AlgorithmicNode]:
        """Return leaf nodes that are NOT atomic (validation failures)."""
        parent_ids = {e.source_id for e in self.edges}
        child_ids = {e.target_id for e in self.edges}
        all_ids = {n.node_id for n in self.nodes}

        # Leaves: nodes that have no children (not a source in any edge,
        # or have no entries in children list)
        leaves = []
        for node in self.nodes:
            if not node.children and node.node_id not in parent_ids:
                if node.status != NodeStatus.ATOMIC:
                    leaves.append(node)
        return leaves

    def is_complete(self) -> bool:
        """Check that all leaf nodes are atomic."""
        return len(self.non_atomic_leaves()) == 0


def export_cdg(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
    *,
    goal: str = "",
    paradigm: str = "",
) -> CDGExport:
    """Build a CDGExport, validating that all leaves are atomic.

    Raises:
        ValueError: If any leaf node is not atomic.
    """
    cdg = CDGExport(
        nodes=nodes,
        edges=edges,
        metadata={
            "goal": goal,
            "paradigm": paradigm,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "num_nodes": len(nodes),
            "num_edges": len(edges),
        },
    )

    non_atomic = cdg.non_atomic_leaves()
    if non_atomic:
        names = [n.name for n in non_atomic]
        raise ValueError(
            f"CDG has {len(non_atomic)} non-atomic leaf node(s): {names}. "
            "All leaves must be marked atomic before export."
        )

    return cdg


def validate_handoff(cdg: CDGExport) -> list[str]:
    """Check that every atomic leaf is ready for Round 2 handoff.

    Returns a list of human-readable issue strings (empty == valid).
    Checks:
    - Every atomic leaf must have a non-empty ``description``.
    - Every atomic leaf must have a non-empty ``type_signature``.
    - Flags non-atomic leaf nodes (they can't be handed off).
    """
    issues: list[str] = []
    parent_ids = {e.source_id for e in cdg.edges}

    for node in cdg.nodes:
        is_leaf = not node.children and node.node_id not in parent_ids
        if not is_leaf:
            continue

        if node.status == NodeStatus.ATOMIC:
            if not node.description:
                issues.append(
                    f"Atomic leaf '{node.name}' ({node.node_id}) has empty description"
                )
            if not node.type_signature:
                issues.append(
                    f"Atomic leaf '{node.name}' ({node.node_id}) has empty type_signature"
                )
        else:
            issues.append(
                f"Leaf '{node.name}' ({node.node_id}) is {node.status.value}, not atomic"
            )

    return issues


def to_pdg_nodes(
    cdg: CDGExport,
    *,
    prover: Prover = Prover.LEAN4,
    strict: bool = True,
) -> list[PDGNode]:
    """Convert atomic leaf nodes of a CDG into PDGNode objects for the Round 2 Hunter.

    Each atomic leaf becomes a PDGNode with:
    - predicate_id: the node_id
    - statement: the type_signature (or description if no type sig)
    - informal_desc: the node description
    - context: parent chain, concept type, matched primitive

    Args:
        strict: When True (default), runs ``validate_handoff`` and raises
            ``HandoffValidationError`` on any issues. Set to False to skip.

    Raises:
        HandoffValidationError: If strict and the CDG has handoff issues.
        ValueError: If the CDG is not complete (has non-atomic leaves).
    """
    if strict:
        issues = validate_handoff(cdg)
        if issues:
            raise HandoffValidationError(issues)

    if not cdg.is_complete():
        non_atomic = cdg.non_atomic_leaves()
        names = [n.name for n in non_atomic]
        raise ValueError(
            f"Cannot convert incomplete CDG. Non-atomic leaves: {names}"
        )

    pdg_nodes: list[PDGNode] = []
    for node in cdg.leaf_nodes():
        statement = node.type_signature or node.description
        context = {
            "concept_type": node.concept_type.value,
            "depth": str(node.depth),
        }
        if node.parent_id:
            context["parent_id"] = node.parent_id
        if node.matched_primitive:
            context["matched_primitive"] = node.matched_primitive
        if node.critic_notes:
            context["critic_notes"] = node.critic_notes

        pdg_node = PDGNode(
            predicate_id=node.node_id,
            statement=statement,
            informal_desc=node.description,
            prover=prover,
            context=context,
        )
        pdg_nodes.append(pdg_node)

    return pdg_nodes


def save_json(cdg: CDGExport, path: str | Path) -> None:
    """Save a CDGExport to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cdg.model_dump(), f, indent=2)


def load_json(path: str | Path) -> CDGExport:
    """Load a CDGExport from a JSON file."""
    path = Path(path)
    with open(path) as f:
        data = json.load(f)
    return CDGExport.model_validate(data)

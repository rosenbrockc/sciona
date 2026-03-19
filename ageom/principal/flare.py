"""Dead-End Flare protocol: extract and serialize frozen optimization state."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class BestStructure(BaseModel):
    """Structural summary of the best CDG variant found."""

    node_count: int = 0
    edge_count: int = 0
    topo_hash: str = ""
    primitive_signature: str = ""
    atomic_primitives: list[str] = Field(default_factory=list)


class FlarePayload(BaseModel):
    """Dead-End Flare: frozen state from an exhausted optimization run.

    PRIVACY: Omits gradient scores, sources.yml paths, embeddings,
    UCB scores, and trial-level performance data.
    """

    goal: str
    objective: str
    execution_metric: str
    domain_tags: list[str] = Field(default_factory=list)
    best_metric_value: float
    metric_name: str
    max_graph_nodes: int = 0
    max_graph_edges: int = 0
    atoms_tried: list[str] = Field(default_factory=list)
    best_structure: BestStructure = Field(default_factory=BestStructure)
    concept_types: list[str] = Field(default_factory=list)


def generate_flare(
    final_state: dict[str, Any],
    *,
    domain_tags: list[str] | None = None,
) -> FlarePayload:
    """Extract a flare from a completed optimization final state dict.

    Parameters
    ----------
    final_state
        The dict returned by ``graph.ainvoke()``, containing keys like
        ``goal``, ``metric``, ``best_loss``, ``trial_history``, and ``cdg``.
    domain_tags
        Optional domain labels the user wants to attach (e.g. ``["crystallography"]``).

    PRIVACY: Omits gradient scores, sources.yml paths, embeddings,
    UCB scores, trial-level performance data.
    """
    history: list[dict[str, Any]] = final_state.get("trial_history", [])
    metric = final_state.get("metric")
    metric_name = metric.value if hasattr(metric, "value") else str(metric or "")

    # Best structure: trial with lowest loss
    best_entry: dict[str, Any] | None = None
    for entry in history:
        if best_entry is None or entry.get("loss", float("inf")) < best_entry.get(
            "loss", float("inf")
        ):
            best_entry = entry

    best_structure = BestStructure()
    if best_entry:
        structure = best_entry.get("structure", {})
        best_structure = BestStructure(
            node_count=structure.get("node_count", 0),
            edge_count=structure.get("edge_count", 0),
            topo_hash=structure.get("topo_hash", ""),
            primitive_signature=structure.get("primitive_signature", ""),
            atomic_primitives=sorted(
                set(structure.get("atomic_primitives", {}).values())
            )
            if isinstance(structure.get("atomic_primitives"), dict)
            else list(structure.get("atomic_primitives", [])),
        )

    # Deduplicated atoms across all trials
    atoms_tried: set[str] = set()
    max_nodes = 0
    max_edges = 0
    for entry in history:
        structure = entry.get("structure", {})
        primitives = structure.get("atomic_primitives", {})
        if isinstance(primitives, dict):
            atoms_tried.update(primitives.values())
        elif isinstance(primitives, list):
            atoms_tried.update(primitives)
        max_nodes = max(max_nodes, structure.get("node_count", 0))
        max_edges = max(max_edges, structure.get("edge_count", 0))

    # Concept types from CDG nodes if available
    concept_types: list[str] = []
    cdg = final_state.get("cdg")
    if cdg is not None and hasattr(cdg, "nodes"):
        seen: set[str] = set()
        for node in cdg.nodes:
            ct = getattr(node, "concept_type", None)
            if ct is not None:
                val = ct.value if hasattr(ct, "value") else str(ct)
                if val and val not in seen:
                    seen.add(val)
                    concept_types.append(val)

    return FlarePayload(
        goal=final_state.get("goal", ""),
        objective=metric_name,
        execution_metric=metric_name,
        domain_tags=domain_tags or [],
        best_metric_value=final_state.get("best_loss", float("inf")),
        metric_name=metric_name,
        max_graph_nodes=max_nodes,
        max_graph_edges=max_edges,
        atoms_tried=sorted(atoms_tried),
        best_structure=best_structure,
        concept_types=concept_types,
    )


def write_flare_config(payload: FlarePayload, output_path: Path) -> Path:
    """Serialize *payload* to a YAML config file (hand-rolled, no pyyaml dep).

    Returns the resolved output path.
    """
    output_path = Path(output_path)
    lines: list[str] = [
        "# Dead-End Flare — frozen optimization state",
        f"goal: {_yaml_quote(payload.goal)}",
        f"objective: {_yaml_quote(payload.objective)}",
        f"execution_metric: {_yaml_quote(payload.execution_metric)}",
        f"best_metric_value: {payload.best_metric_value}",
        f"metric_name: {_yaml_quote(payload.metric_name)}",
        f"max_graph_nodes: {payload.max_graph_nodes}",
        f"max_graph_edges: {payload.max_graph_edges}",
    ]

    lines.append("domain_tags:")
    for tag in payload.domain_tags:
        lines.append(f"  - {_yaml_quote(tag)}")

    lines.append("atoms_tried:")
    for atom in payload.atoms_tried:
        lines.append(f"  - {_yaml_quote(atom)}")

    lines.append("concept_types:")
    for ct in payload.concept_types:
        lines.append(f"  - {_yaml_quote(ct)}")

    lines.append("best_structure:")
    lines.append(f"  node_count: {payload.best_structure.node_count}")
    lines.append(f"  edge_count: {payload.best_structure.edge_count}")
    lines.append(f"  topo_hash: {_yaml_quote(payload.best_structure.topo_hash)}")
    lines.append(
        f"  primitive_signature: {_yaml_quote(payload.best_structure.primitive_signature)}"
    )
    lines.append("  atomic_primitives:")
    for prim in payload.best_structure.atomic_primitives:
        lines.append(f"    - {_yaml_quote(prim)}")

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output_path


def _yaml_quote(value: str) -> str:
    """Quote a string for safe YAML scalar emission."""
    if not value:
        return '""'
    # Quote if it contains characters that could be misinterpreted
    needs_quote = any(
        c in value for c in (":", "#", "'", '"', "{", "}", "[", "]", ",", "\n")
    )
    if needs_quote or value.strip() != value:
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    return value

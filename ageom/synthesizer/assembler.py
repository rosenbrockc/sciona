"""Core assembly logic: CDG + MatchResults -> compilable skeleton file."""

from __future__ import annotations

import re
from datetime import datetime, timezone

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, NodeStatus
from ageom.synthesizer.models import AssemblyUnit, GlueEdge, SkeletonFile
from ageom.synthesizer.toposort import toposort_nodes
from ageom.types import MatchResult, Prover


class AssemblyError(Exception):
    """Raised when assembly fails due to missing or invalid inputs."""


def sanitize_name(name: str) -> str:
    """Convert a human-readable name to a valid Lean/Coq identifier."""
    # Replace non-alphanumeric with underscores
    s = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Collapse consecutive underscores
    s = re.sub(r"_+", "_", s)
    # Strip leading/trailing underscores
    s = s.strip("_")
    # Lowercase
    s = s.lower()
    # Ensure it starts with a letter
    if s and not s[0].isalpha():
        s = "n_" + s
    return s or "unnamed"


class Assembler:
    """Assembles a CDG and match results into a compilable skeleton."""

    def __init__(self, prover: Prover | str) -> None:
        if isinstance(prover, str):
            self._prover = Prover(prover)
        else:
            self._prover = prover

    def assemble(
        self,
        cdg: CDGExport,
        match_results: list[MatchResult],
    ) -> SkeletonFile:
        """Build a SkeletonFile from a CDG and its match results."""
        # Index match results by predicate_id (= node_id)
        match_map: dict[str, MatchResult] = {}
        for mr in match_results:
            match_map[mr.pdg_node.predicate_id] = mr

        # Validate: every atomic leaf must have a successful match
        atomic_leaves = [n for n in cdg.nodes if n.status == NodeStatus.ATOMIC]
        missing: list[str] = []
        for leaf in atomic_leaves:
            mr = match_map.get(leaf.node_id)
            if mr is None or not mr.success:
                missing.append(leaf.name)
        if missing:
            raise AssemblyError(
                f"Missing verified matches for atomic leaves: {missing}"
            )

        # Build AssemblyUnits for atomic leaves
        units: list[AssemblyUnit] = []
        glue_node_ids: set[str] = set()
        for edge in cdg.edges:
            if edge.requires_glue:
                glue_node_ids.add(edge.target_id)

        for leaf in atomic_leaves:
            mr = match_map[leaf.node_id]
            assert mr.verified_match is not None
            decl = mr.verified_match.candidate.declaration
            units.append(
                AssemblyUnit(
                    node_id=leaf.node_id,
                    name=leaf.name,
                    declaration_name=decl.name,
                    type_signature=decl.type_signature or leaf.type_signature,
                    raw_code=decl.raw_code,
                    inputs=leaf.inputs,
                    outputs=leaf.outputs,
                    requires_glue=leaf.node_id in glue_node_ids,
                )
            )

        # Build GlueEdges
        glue_edges: list[GlueEdge] = []
        for edge in cdg.edges:
            glue_edges.append(
                GlueEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    output_name=edge.output_name,
                    input_name=edge.input_name,
                    source_type=edge.source_type,
                    target_type=edge.target_type,
                    cast_expr="",  # Phase 2 fills this
                )
            )

        # Topological sort for emission order
        sorted_ids = toposort_nodes(cdg.nodes, cdg.edges)

        # Reorder units by topological order
        unit_map = {u.node_id: u for u in units}
        sorted_units = [unit_map[nid] for nid in sorted_ids if nid in unit_map]

        # Find root nodes (decomposed, not atomic)
        root_nodes = [
            n for n in cdg.nodes if n.status == NodeStatus.DECOMPOSED
        ]

        # Generate source code
        metadata = dict(cdg.metadata) if cdg.metadata else {}
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

        if self._prover == Prover.LEAN4:
            source, sorry_count = self._emit_lean4(
                sorted_units, glue_edges, root_nodes, metadata
            )
        else:
            source, sorry_count = self._emit_coq(
                sorted_units, glue_edges, root_nodes, metadata
            )

        return SkeletonFile(
            prover=self._prover.value,
            source_code=source,
            units=sorted_units,
            glue_edges=glue_edges,
            sorry_count=sorry_count,
            metadata=metadata,
        )

    def _emit_lean4(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
    ) -> tuple[str, int]:
        """Generate Lean 4 source. Returns (source_code, sorry_count)."""
        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("/-!")
        lines.append(f"  AGEO-Matcher Skeleton")
        goal = metadata.get("goal", "")
        if goal:
            lines.append(f"  Goal: {goal}")
        lines.append(f"  Generated: {metadata.get('timestamp', '')}")
        lines.append("-/")
        lines.append("")
        lines.append("import Mathlib")
        lines.append("")

        # Emit atomic leaf definitions
        for unit in units:
            sname = sanitize_name(unit.name)
            lines.append(f"-- Node: {unit.name} ({unit.node_id})")
            lines.append(f"#check @{unit.declaration_name}")
            if unit.type_signature:
                lines.append(
                    f"noncomputable def {sname} : {unit.type_signature} :="
                )
                lines.append(f"  @{unit.declaration_name}")
            else:
                lines.append(f"noncomputable def {sname} := @{unit.declaration_name}")
            lines.append("")

        # Emit composition stubs for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"-- Composition: {root.name} ({root.node_id})")
                lines.append(
                    f"theorem {rname}_composition : {root.type_signature} := by"
                )
                lines.append("  sorry")
                sorry_count += 1
                lines.append("")

        return "\n".join(lines), sorry_count

    def _emit_coq(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
    ) -> tuple[str, int]:
        """Generate Coq source. Returns (source_code, sorry_count)."""
        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("(*")
        lines.append(f"  AGEO-Matcher Skeleton")
        goal = metadata.get("goal", "")
        if goal:
            lines.append(f"  Goal: {goal}")
        lines.append(f"  Generated: {metadata.get('timestamp', '')}")
        lines.append("*)")
        lines.append("")

        # Emit atomic leaf definitions
        for unit in units:
            sname = sanitize_name(unit.name)
            lines.append(f"(* Node: {unit.name} ({unit.node_id}) *)")
            lines.append(f"Check @{unit.declaration_name}.")
            if unit.type_signature:
                lines.append(
                    f"Definition {sname} : {unit.type_signature} := @{unit.declaration_name}."
                )
            else:
                lines.append(
                    f"Definition {sname} := @{unit.declaration_name}."
                )
            lines.append("")

        # Emit composition stubs
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"(* Composition: {root.name} ({root.node_id}) *)")
                lines.append(
                    f"Lemma {rname}_composition : {root.type_signature}."
                )
                lines.append("Proof. Admitted.")
                sorry_count += 1
                lines.append("")

        return "\n".join(lines), sorry_count

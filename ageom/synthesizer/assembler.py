"""Core assembly logic: CDG + MatchResults -> compilable skeleton file."""

from __future__ import annotations

import ast
import re
from datetime import datetime, timezone

from ageom.architect.handoff import CDGExport
from ageom.architect.models import AlgorithmicNode, NodeStatus
from ageom.synthesizer.models import AssemblyUnit, GlueEdge, SkeletonFile
from ageom.synthesizer.toposort import toposort_nodes
from ageom.types import MatchResult, Prover

# ---------------------------------------------------------------------------
# Telemetry helper emitted into instrumented Python skeletons
# ---------------------------------------------------------------------------

_PYTHON_TELEMETRY_HELPER: list[str] = [
    "",
    "_AGEOM_TRACE_PATH = 'trace.jsonl'",
    "",
    "",
    "def _ageom_probe(node_id: str, fn):",
    '    """Execute *fn* and append a JSON-lines telemetry record."""',
    "    tracemalloc.start()",
    "    t0 = time.perf_counter()",
    "    try:",
    "        result = fn()",
    "    finally:",
    "        elapsed_ms = (time.perf_counter() - t0) * 1000.0",
    "        _, peak = tracemalloc.get_traced_memory()",
    "        tracemalloc.stop()",
    "        record = {",
    '            "node_id": node_id,',
    '            "execution_time_ms": elapsed_ms,',
    '            "peak_memory_bytes": peak,',
    "        }",
    "        with open(_AGEOM_TRACE_PATH, 'a') as _f:",
    "            _f.write(json.dumps(record) + '\\n')",
    "    return result",
]


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


def _python_annotation_expr(type_desc: str) -> str:
    """Return a syntactically valid Python annotation expression.

    Conceptual labels from the architect catalog often contain spaces
    (for example ``filter specification``). Those are useful to preserve,
    but they are not valid Python expressions unless quoted.
    """
    annotation = type_desc.strip()
    if not annotation:
        return "object"
    try:
        ast.parse(annotation, mode="eval")
    except SyntaxError:
        return repr(annotation)
    return annotation


def _split_top_level(text: str, delimiter: str) -> list[str]:
    """Split *text* on a delimiter, ignoring nested bracket scopes."""
    parts: list[str] = []
    start = 0
    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == delimiter and depth == 0:
            parts.append(text[start:idx])
            start = idx + 1
    parts.append(text[start:])
    return parts


def _find_top_level(text: str, needle: str) -> int:
    """Find a top-level delimiter position, ignoring nested bracket scopes."""
    depth = 0
    for idx, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth = max(0, depth - 1)
        elif char == needle and depth == 0:
            return idx
    return -1


def _sanitize_python_param_annotation(param: str) -> str:
    """Quote conceptual Python annotations inside a single function param."""
    colon = _find_top_level(param, ":")
    if colon < 0:
        return param

    left = param[:colon].rstrip()
    right = param[colon + 1 :]
    default_idx = _find_top_level(right, "=")
    if default_idx >= 0:
        annotation = right[:default_idx].strip()
        default = right[default_idx:]
    else:
        annotation = right.strip()
        default = ""

    sanitized = _python_annotation_expr(annotation)
    rebuilt = f"{left}: {sanitized}"
    if default:
        rebuilt += default
    return rebuilt


def sanitize_python_source_annotations(source: str) -> str:
    """Rewrite invalid Python def annotations into syntactically valid forms."""
    lines = source.splitlines()
    sanitized_lines: list[str] = []

    for line in lines:
        if not re.match(r"^\s*def\s+\w+\s*\(", line):
            sanitized_lines.append(line)
            continue

        open_paren = line.find("(")
        if open_paren < 0:
            sanitized_lines.append(line)
            continue

        depth = 0
        close_paren = -1
        for idx in range(open_paren, len(line)):
            char = line[idx]
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
                if depth == 0:
                    close_paren = idx
                    break
        if close_paren < 0:
            sanitized_lines.append(line)
            continue

        prefix = line[: open_paren + 1]
        params_text = line[open_paren + 1 : close_paren]
        suffix = line[close_paren + 1 :]

        sanitized_params = ",".join(
            _sanitize_python_param_annotation(param)
            for param in _split_top_level(params_text, ",")
        )

        sanitized_suffix = suffix
        arrow_idx = suffix.find("->")
        colon_idx = suffix.rfind(":")
        if arrow_idx >= 0 and colon_idx > arrow_idx:
            annotation = suffix[arrow_idx + 2 : colon_idx].strip()
            sanitized_return = _python_annotation_expr(annotation)
            sanitized_suffix = f" -> {sanitized_return}{suffix[colon_idx:]}"

        sanitized_lines.append(f"{prefix}{sanitized_params}){sanitized_suffix}")

    result = "\n".join(sanitized_lines)
    if source.endswith("\n"):
        result += "\n"
    return result


def _infer_cast(source_type: str, target_type: str) -> str:
    """Infer a cast expression for a GlueEdge between source and target types.

    Returns an empty string when no cast is needed, or a code snippet
    appropriate for the type transformation.
    """
    src = source_type.strip()
    tgt = target_type.strip()

    if not src or not tgt:
        return ""

    # Same type -> identity (no cast)
    if src == tgt:
        return ""

    # ndarray shape changes -> np.reshape()
    if "ndarray" in src and "ndarray" in tgt:
        return "np.reshape({src}, {tgt_shape})"

    # Tuple destructure: (b, a) -> individual components
    if src.startswith("(") and "," in src and not tgt.startswith("("):
        return "({src})[0]"

    # Lean coercions
    if "Equiv" in src or "Equiv" in tgt:
        return "Equiv.toFun"

    return ""


class Assembler:
    """Assembles a CDG and match results into a compilable skeleton."""

    def __init__(self, prover: Prover | str, *, with_telemetry: bool = False) -> None:
        if isinstance(prover, str):
            self._prover = Prover(prover)
        else:
            self._prover = prover
        self._with_telemetry = with_telemetry

    def assemble(
        self,
        cdg: CDGExport,
        match_results: list[MatchResult],
        *,
        with_telemetry: bool | None = None,
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

        # Build GlueEdges with cast inference
        glue_edges: list[GlueEdge] = []
        for edge in cdg.edges:
            cast = _infer_cast(edge.source_type, edge.target_type)
            glue_edges.append(
                GlueEdge(
                    source_id=edge.source_id,
                    target_id=edge.target_id,
                    output_name=edge.output_name,
                    input_name=edge.input_name,
                    source_type=edge.source_type,
                    target_type=edge.target_type,
                    cast_expr=cast,
                )
            )

        # Topological sort for emission order
        sorted_ids = toposort_nodes(cdg.nodes, cdg.edges)

        # Reorder units by topological order
        unit_map = {u.node_id: u for u in units}
        sorted_units = [unit_map[nid] for nid in sorted_ids if nid in unit_map]

        # Find root nodes (decomposed, not atomic)
        root_nodes = [n for n in cdg.nodes if n.status == NodeStatus.DECOMPOSED]

        # Generate source code
        metadata = dict(cdg.metadata) if cdg.metadata else {}
        metadata["timestamp"] = datetime.now(timezone.utc).isoformat()

        telemetry = (
            with_telemetry if with_telemetry is not None else self._with_telemetry
        )

        if self._prover == Prover.LEAN4:
            source, sorry_count = self._emit_lean4(
                sorted_units,
                glue_edges,
                root_nodes,
                metadata,
                telemetry=telemetry,
            )
        elif self._prover == Prover.PYTHON:
            source, sorry_count = self._emit_python(
                sorted_units,
                glue_edges,
                root_nodes,
                metadata,
                telemetry=telemetry,
            )
        else:
            source, sorry_count = self._emit_coq(
                sorted_units,
                glue_edges,
                root_nodes,
                metadata,
                telemetry=telemetry,
            )

        return SkeletonFile(
            prover=self._prover.value,
            source_code=source,
            units=sorted_units,
            glue_edges=glue_edges,
            sorry_count=sorry_count,
            metadata=metadata,
        )

    # ------------------------------------------------------------------
    # Composition helpers
    # ------------------------------------------------------------------

    def _compose_python(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root: AlgorithmicNode,
    ) -> list[str]:
        """Generate Python composition body for a root node.

        Walks glue edges in topological order and emits variable assignments
        connecting source outputs to target inputs.
        """
        unit_map = {u.node_id: u for u in units}

        # Edges relevant to this root's children
        child_ids = set(root.children) if root.children else {u.node_id for u in units}
        relevant_edges = [
            e
            for e in glue_edges
            if e.source_id in child_ids or e.target_id in child_ids
        ]

        if not relevant_edges and not units:
            return [
                '    raise NotImplementedError("TODO: compose {}")'.format(root.name)
            ]

        lines: list[str] = []

        # First, call each atomic unit in topological order
        called: set[str] = set()
        for unit in units:
            if unit.node_id not in child_ids:
                continue
            sname = sanitize_name(unit.name)
            # Determine input arguments from edges or function inputs
            args: list[str] = []
            for inp in unit.inputs:
                # Check if an edge provides this input
                edge_for_inp = next(
                    (
                        e
                        for e in relevant_edges
                        if e.target_id == unit.node_id and e.input_name == inp.name
                    ),
                    None,
                )
                if edge_for_inp:
                    src_unit = unit_map.get(edge_for_inp.source_id)
                    if src_unit:
                        src_var = sanitize_name(src_unit.name) + "_result"
                        if edge_for_inp.cast_expr:
                            args.append(
                                f"# cast: {edge_for_inp.cast_expr}\n    {src_var}"
                            )
                        else:
                            args.append(src_var)
                    else:
                        args.append(inp.name)
                else:
                    args.append(inp.name)

            args_str = ", ".join(args)
            lines.append(f"    {sname}_result = {sname}({args_str})")
            called.add(unit.node_id)

        # Return the last unit's result
        if called:
            last_unit = [u for u in units if u.node_id in called]
            if last_unit:
                last_name = sanitize_name(last_unit[-1].name) + "_result"
                lines.append(f"    return {last_name}")
        else:
            lines.append("    # TODO: compose -- no atomic children resolved")
            lines.append(
                '    raise NotImplementedError("compose {}")'.format(root.name)
            )

        return lines

    def _compose_lean4(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root: AlgorithmicNode,
    ) -> list[str]:
        """Generate Lean 4 composition proof body for a root node.

        Uses direct term composition (f . g) or calc chains instead of sorry.
        """
        child_ids = set(root.children) if root.children else {u.node_id for u in units}
        relevant_units = [u for u in units if u.node_id in child_ids]

        if not relevant_units:
            return [
                "  -- TODO: compose {} -- no atomic children resolved".format(
                    root.name
                ),
                "  sorry",
            ]

        lines: list[str] = []

        if len(relevant_units) == 1:
            # Single child: direct application
            u = relevant_units[0]
            sname = sanitize_name(u.name)
            lines.append(f"  exact {sname}")
        else:
            # Multiple children: compose via term application
            names = [sanitize_name(u.name) for u in relevant_units]

            # Check for glue edges needing casts
            has_casts = any(
                e.cast_expr
                for e in glue_edges
                if e.source_id in child_ids or e.target_id in child_ids
            )
            if has_casts:
                for e in glue_edges:
                    if e.cast_expr and (
                        e.source_id in child_ids or e.target_id in child_ids
                    ):
                        lines.append(f"  -- GLUE: {e.source_type} -> {e.target_type}")
                lines.append(f"  exact ({' ∘ '.join(reversed(names))})")
            else:
                lines.append(f"  exact ({' ∘ '.join(reversed(names))})")

        return lines

    def _compose_coq(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root: AlgorithmicNode,
    ) -> list[str]:
        """Generate Coq composition proof body for a root node.

        Uses exact/apply chains instead of Admitted.
        """
        child_ids = set(root.children) if root.children else {u.node_id for u in units}
        relevant_units = [u for u in units if u.node_id in child_ids]

        if not relevant_units:
            return [
                "  (* TODO: compose {} -- no atomic children resolved *)".format(
                    root.name
                ),
                "  Admitted.",
            ]

        if len(relevant_units) == 1:
            u = relevant_units[0]
            sname = sanitize_name(u.name)
            return [f"Proof. exact {sname}. Qed."]
        else:
            names = [sanitize_name(u.name) for u in relevant_units]
            apply_chain = " ".join(f"apply {n}." for n in names)
            return [f"Proof. {apply_chain} Qed."]

    # ------------------------------------------------------------------
    # Prover-specific emitters
    # ------------------------------------------------------------------

    def _emit_lean4(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
        *,
        telemetry: bool = False,
    ) -> tuple[str, int]:
        """Generate Lean 4 source. Returns (source_code, sorry_count)."""
        if telemetry:
            raise NotImplementedError(
                "Lean 4 telemetry instrumentation is not yet implemented. "
                "Lean's `timeit` tactic and custom `IO.monoMsNow` wrappers "
                "will be integrated here."
            )

        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("/-!")
        lines.append("  AGEO-Matcher Skeleton")
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
                lines.append(f"noncomputable def {sname} : {unit.type_signature} :=")
                lines.append(f"  @{unit.declaration_name}")
            else:
                lines.append(f"noncomputable def {sname} := @{unit.declaration_name}")
            lines.append("")

        # Emit composition for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"-- Composition: {root.name} ({root.node_id})")
                lines.append(
                    f"theorem {rname}_composition : {root.type_signature} := by"
                )
                composition = self._compose_lean4(units, glue_edges, root)
                lines.extend(composition)
                # Count sorrys in generated composition
                for line in composition:
                    if "sorry" in line and not line.strip().startswith("--"):
                        sorry_count += 1
                lines.append("")

        return "\n".join(lines), sorry_count

    def _emit_python(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
        *,
        telemetry: bool = False,
    ) -> tuple[str, int]:
        """Generate Python source. Returns (source_code, sorry_count)."""
        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("from __future__ import annotations")
        lines.append("")
        lines.append('"""')
        lines.append("AGEO-Matcher Skeleton")
        goal = metadata.get("goal", "")
        if goal:
            lines.append(f"Goal: {goal}")
        lines.append(f"Generated: {metadata.get('timestamp', '')}")
        lines.append('"""')
        lines.append("")

        # Imports
        lines.append("import icontract")

        # Infer imports from declaration names
        imports_seen: set[str] = {"icontract"}
        for unit in units:
            if "." in unit.declaration_name:
                module = unit.declaration_name.rsplit(".", 1)[0]
                top_level = module.split(".")[0]
                if top_level not in imports_seen:
                    imports_seen.add(top_level)
                    lines.append(f"import {top_level}")

        # Common scientific imports if not already present
        for pkg in ("numpy", "scipy"):
            if pkg not in imports_seen:
                imports_seen.add(pkg)
                lines.append(f"import {pkg}")

        # Telemetry imports
        if telemetry:
            lines.append("import json")
            lines.append("import time")
            lines.append("import tracemalloc")

        lines.append("")

        # Telemetry helper
        if telemetry:
            lines.extend(_PYTHON_TELEMETRY_HELPER)
            lines.append("")

        lines.append("")

        # Emit atomic leaf definitions
        for unit in units:
            sname = sanitize_name(unit.name)
            lines.append(f"# Node: {unit.name} ({unit.node_id})")

            # Build function signature from unit inputs/outputs
            params: list[str] = []
            for inp in unit.inputs:
                if inp.type_desc:
                    params.append(f"{inp.name}: {_python_annotation_expr(inp.type_desc)}")
                else:
                    params.append(inp.name)

            ret_type = ""
            if unit.outputs:
                ret_type = _python_annotation_expr(unit.outputs[0].type_desc)

            param_str = ", ".join(params)
            ret_str = f" -> {ret_type}" if ret_type else ""
            lines.append(f"def {sname}({param_str}){ret_str}:")

            if unit.type_signature:
                lines.append(f'    """Type: {unit.type_signature}"""')

            if telemetry:
                arg_str = ", ".join(inp.name for inp in unit.inputs)
                call_expr = f"{unit.declaration_name}({arg_str})"
                lines.append(
                    f"    return _ageom_probe({unit.node_id!r}, lambda: {call_expr})"
                )
            else:
                lines.append(
                    f"    return {unit.declaration_name}({', '.join(inp.name for inp in unit.inputs)})"
                )
            lines.append("")
            lines.append("")

        # Emit composition for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"# Composition: {root.name} ({root.node_id})")

                # Build composition function with appropriate params from root inputs
                params = []
                for inp in root.inputs:
                    if inp.type_desc:
                        params.append(f"{inp.name}: {_python_annotation_expr(inp.type_desc)}")
                    else:
                        params.append(inp.name)
                param_str = ", ".join(params) if params else ""

                ret_type = ""
                if root.outputs:
                    ret_type = _python_annotation_expr(root.outputs[0].type_desc)
                ret_str = f" -> {ret_type}" if ret_type else ""

                lines.append(f"def {rname}_composition({param_str}){ret_str}:")
                lines.append(f'    """Compose: {root.type_signature}"""')

                composition = self._compose_python(units, glue_edges, root)
                lines.extend(composition)
                # Count NotImplementedError stubs
                for line in composition:
                    if "NotImplementedError" in line:
                        sorry_count += 1
                lines.append("")
                lines.append("")

        return "\n".join(lines), sorry_count

    def _emit_coq(
        self,
        units: list[AssemblyUnit],
        glue_edges: list[GlueEdge],
        root_nodes: list[AlgorithmicNode],
        metadata: dict,
        *,
        telemetry: bool = False,
    ) -> tuple[str, int]:
        """Generate Coq source. Returns (source_code, sorry_count)."""
        if telemetry:
            raise NotImplementedError(
                "Coq telemetry instrumentation is not yet implemented. "
                "Coq's `Time` vernacular command will be integrated here."
            )
        lines: list[str] = []
        sorry_count = 0

        # Header
        lines.append("(*")
        lines.append("  AGEO-Matcher Skeleton")
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
                lines.append(f"Definition {sname} := @{unit.declaration_name}.")
            lines.append("")

        # Emit composition for root/decomposed nodes
        for root in root_nodes:
            rname = sanitize_name(root.name)
            if root.type_signature:
                lines.append(f"(* Composition: {root.name} ({root.node_id}) *)")
                lines.append(f"Lemma {rname}_composition : {root.type_signature}.")
                composition = self._compose_coq(units, glue_edges, root)
                lines.extend(composition)
                for line in composition:
                    if "Admitted" in line:
                        sorry_count += 1
                lines.append("")

        return "\n".join(lines), sorry_count

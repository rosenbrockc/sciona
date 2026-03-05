"""Deterministic post-processing for Architect decomposition payloads.

The LLM proposes conceptual sub-nodes (names/descriptions and optional hints).
This module deterministically synthesizes operational details:
- concept type fallback/inference,
- IO specs,
- atomic status and primitive binding,
- edge wiring and type propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
import uuid
from typing import Any

from ageom.architect.catalog import PrimitiveCatalog
from ageom.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)


@dataclass
class DeterministicDecomposeResult:
    """Deterministic decomposition artifacts."""

    nodes: list[AlgorithmicNode]
    edges: list[DependencyEdge]


_KEYWORD_CONCEPTS: list[tuple[tuple[str, ...], ConceptType]] = [
    (("sort", "ordered", "rank"), ConceptType.SORTING),
    (("search", "lookup", "find"), ConceptType.SEARCHING),
    (("split", "partition", "divide"), ConceptType.DIVIDE_AND_CONQUER),
    (("merge", "combine", "aggregate"), ConceptType.DATA_ASSEMBLY),
    (("graph", "node", "edge", "path"), ConceptType.GRAPH_TRAVERSAL),
    (("threshold", "hysteresis", "state machine"), ConceptType.SEQUENTIAL_FILTER),
    (("smooth", "filter", "denoise"), ConceptType.SIGNAL_FILTER),
    (("transform", "fft", "wavelet"), ConceptType.SIGNAL_TRANSFORM),
    (("sample", "proposal", "mcmc", "nuts", "leapfrog"), ConceptType.SAMPLER),
    (("prior", "posterior", "conjugate", "bayes"), ConceptType.CONJUGATE_UPDATE),
    (("message", "belief", "factor"), ConceptType.MESSAGE_PASSING),
    (("loss", "gradient", "backprop"), ConceptType.NEURAL_NETWORK),
]

_CONCEPTUAL_FALLBACKS: dict[ConceptType, list[dict[str, str]]] = {
    ConceptType.SIGNAL_FILTER: [
        {
            "name": "Parse Filter Requirements",
            "description": (
                "Extract sample-rate, passband/stopband, attenuation, ripple, and "
                "implementation constraints from the specification."
            ),
            "concept_type": ConceptType.DATA_ASSEMBLY.value,
            "matched_primitive_hint": "parse_filter_spec",
        },
        {
            "name": "Select Filter Family",
            "description": (
                "Choose FIR or IIR topology and design method based on phase, "
                "stability, and compute constraints."
            ),
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "choose_filter_topology",
        },
        {
            "name": "Synthesize Coefficients",
            "description": (
                "Generate candidate coefficients from the selected method and target "
                "frequency response."
            ),
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "design_filter_coefficients",
        },
        {
            "name": "Validate and Finalize Coefficients",
            "description": (
                "Check frequency-response compliance and stability, then return the "
                "final coefficient vector."
            ),
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "validate_filter_response",
        },
    ],
}

_GENERIC_FALLBACK: list[dict[str, str]] = [
    {
        "name": "Interpret Requirements",
        "description": "Normalize inputs, constraints, and success criteria for this node.",
    },
    {
        "name": "Compute Core Transformation",
        "description": "Perform the main transformation required to reach the target output.",
    },
    {
        "name": "Validate and Return Result",
        "description": "Verify constraints and emit the final result in the expected shape.",
    },
]


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _safe_iospec_list(raw: Any) -> list[IOSpec]:
    if not isinstance(raw, list):
        return []
    rows: list[IOSpec] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        if not name:
            continue
        type_desc = str(item.get("type_desc", "Any")).strip() or "Any"
        constraints = str(item.get("constraints", "")).strip()
        rows.append(IOSpec(name=name, type_desc=type_desc, constraints=constraints))
    return rows


def _find_primitive(catalog: PrimitiveCatalog, name: str) -> Any:
    if not name:
        return None
    direct = catalog.get(name)
    if direct is not None:
        return direct
    target = _norm(name)
    for prim in catalog.all_primitives():
        if _norm(prim.name) == target:
            return prim
    return None


def _infer_concept_type(
    spec: dict[str, Any],
    *,
    parent_type: ConceptType,
    catalog: PrimitiveCatalog,
) -> ConceptType:
    concept_str = str(spec.get("concept_type", "")).strip()
    if concept_str:
        for ct in ConceptType:
            if ct.value == concept_str:
                return ct

    primitive_hint = str(
        spec.get("matched_primitive")
        or spec.get("matched_primitive_hint")
        or spec.get("atomic_hint")
        or ""
    ).strip()
    prim = _find_primitive(catalog, primitive_hint)
    if prim is not None:
        return prim.category

    name = str(spec.get("name", ""))
    prim_from_name = _find_primitive(catalog, name)
    if prim_from_name is not None:
        return prim_from_name.category

    text = f"{name} {str(spec.get('description', ''))}".lower()
    for words, concept in _KEYWORD_CONCEPTS:
        if any(w in text for w in words):
            return concept
    return parent_type


def _default_inputs(parent: AlgorithmicNode) -> list[IOSpec]:
    if parent.inputs:
        return [IOSpec(name=i.name, type_desc=i.type_desc, constraints=i.constraints) for i in parent.inputs]
    return [IOSpec(name="input", type_desc="Any")]


def _default_outputs(parent: AlgorithmicNode) -> list[IOSpec]:
    if parent.outputs:
        return [IOSpec(name=o.name, type_desc=o.type_desc, constraints=o.constraints) for o in parent.outputs]
    return [IOSpec(name="result", type_desc="Any")]


def _ensure_ports(
    spec: dict[str, Any],
    *,
    index: int,
    total: int,
    parent: AlgorithmicNode,
) -> tuple[list[IOSpec], list[IOSpec]]:
    inputs = _safe_iospec_list(spec.get("inputs"))
    outputs = _safe_iospec_list(spec.get("outputs"))

    if total == 1:
        if not inputs:
            inputs = _default_inputs(parent)
        if not outputs:
            outputs = _default_outputs(parent)
        return inputs, outputs

    if not inputs:
        if index == 0:
            inputs = _default_inputs(parent)
        else:
            inputs = [IOSpec(name="input", type_desc="Any")]

    if not outputs:
        if index == total - 1:
            outputs = _default_outputs(parent)
        else:
            outputs = [IOSpec(name="result", type_desc="Any")]

    return inputs, outputs


def _signature_from_ports(inputs: list[IOSpec], outputs: list[IOSpec]) -> str:
    in_sig = ", ".join(i.type_desc or "Any" for i in inputs) or "unit"
    out_sig = ", ".join(o.type_desc or "Any" for o in outputs) or "unit"
    if len(inputs) > 1:
        in_sig = f"({in_sig})"
    if len(outputs) > 1:
        out_sig = f"({out_sig})"
    return f"{in_sig} -> {out_sig}"


def _pick_output_for_target(source: AlgorithmicNode, target: AlgorithmicNode) -> IOSpec:
    if not source.outputs:
        return IOSpec(name="result", type_desc="Any")
    target_inputs = {i.name for i in target.inputs}
    for out in source.outputs:
        if out.name in target_inputs:
            return out
    return source.outputs[0]


def _pick_input_for_output(target: AlgorithmicNode, output_name: str) -> IOSpec:
    if not target.inputs:
        return IOSpec(name="input", type_desc="Any")
    for inp in target.inputs:
        if inp.name == output_name:
            return inp
    return target.inputs[0]


def _edge_tuple(edge: DependencyEdge) -> tuple[str, str, str, str, str, str]:
    return (
        edge.source_id,
        edge.target_id,
        edge.output_name,
        edge.input_name,
        edge.source_type,
        edge.target_type,
    )


def _sanitize_edge_hints(
    nodes: list[AlgorithmicNode],
    edge_hints: list[dict[str, Any]],
) -> list[DependencyEdge]:
    by_norm_name = {_norm(n.name): n for n in nodes}
    seen: set[tuple[str, str, str, str, str, str]] = set()
    result: list[DependencyEdge] = []

    for hint in edge_hints:
        src_name = str(
            hint.get("source_name")
            or hint.get("source")
            or hint.get("from")
            or ""
        ).strip()
        tgt_name = str(
            hint.get("target_name")
            or hint.get("target")
            or hint.get("to")
            or ""
        ).strip()
        if not src_name or not tgt_name:
            continue
        src = by_norm_name.get(_norm(src_name))
        tgt = by_norm_name.get(_norm(tgt_name))
        if src is None or tgt is None or src.node_id == tgt.node_id:
            continue

        preferred_output = str(hint.get("output_name", "")).strip()
        preferred_input = str(hint.get("input_name", "")).strip()

        src_output = _pick_output_for_target(src, tgt)
        if preferred_output and any(o.name == preferred_output for o in src.outputs):
            src_output = next(o for o in src.outputs if o.name == preferred_output)

        tgt_input = _pick_input_for_output(tgt, src_output.name)
        if preferred_input and any(i.name == preferred_input for i in tgt.inputs):
            tgt_input = next(i for i in tgt.inputs if i.name == preferred_input)

        edge = DependencyEdge(
            source_id=src.node_id,
            target_id=tgt.node_id,
            output_name=src_output.name,
            input_name=tgt_input.name,
            source_type=src_output.type_desc,
            target_type=tgt_input.type_desc,
            requires_glue=(
                src_output.type_desc not in {"", "Any"}
                and tgt_input.type_desc not in {"", "Any"}
                and src_output.type_desc != tgt_input.type_desc
            ),
        )
        key = _edge_tuple(edge)
        if key in seen:
            continue
        seen.add(key)
        result.append(edge)

    return result


def _build_chain_edges(nodes: list[AlgorithmicNode]) -> list[DependencyEdge]:
    if len(nodes) <= 1:
        return []
    edges: list[DependencyEdge] = []
    for i in range(len(nodes) - 1):
        source = nodes[i]
        target = nodes[i + 1]
        src_output = _pick_output_for_target(source, target)
        tgt_input = _pick_input_for_output(target, src_output.name)
        edges.append(
            DependencyEdge(
                source_id=source.node_id,
                target_id=target.node_id,
                output_name=src_output.name,
                input_name=tgt_input.name,
                source_type=src_output.type_desc,
                target_type=tgt_input.type_desc,
                requires_glue=(
                    src_output.type_desc not in {"", "Any"}
                    and tgt_input.type_desc not in {"", "Any"}
                    and src_output.type_desc != tgt_input.type_desc
                ),
            )
        )
    return edges


def _collect_edge_hints(parsed: dict[str, Any]) -> list[dict[str, Any]]:
    hints: list[dict[str, Any]] = []
    raw_edges = parsed.get("edges")
    if isinstance(raw_edges, list):
        hints.extend(e for e in raw_edges if isinstance(e, dict))
    raw_flow = parsed.get("flow_hints")
    if isinstance(raw_flow, list):
        hints.extend(e for e in raw_flow if isinstance(e, dict))
    return hints


def _fallback_sub_nodes(parent: AlgorithmicNode) -> list[dict[str, str]]:
    specialized = _CONCEPTUAL_FALLBACKS.get(parent.concept_type)
    if specialized:
        return [dict(item) for item in specialized]
    generic = []
    for item in _GENERIC_FALLBACK:
        row = dict(item)
        row["concept_type"] = parent.concept_type.value
        generic.append(row)
    return generic


def _prepare_raw_sub_nodes(
    raw_subs: Any,
    *,
    parent: AlgorithmicNode,
) -> list[dict[str, Any]]:
    if isinstance(raw_subs, list):
        cleaned = [item for item in raw_subs if isinstance(item, dict)]
    else:
        cleaned = []

    if len(cleaned) >= 2:
        return cleaned

    # If the model returned too little structure, synthesize conceptual
    # decomposition steps deterministically so downstream progress continues.
    fallback = _fallback_sub_nodes(parent)
    if not cleaned:
        return fallback

    existing_norm = {_norm(str(item.get("name", ""))) for item in cleaned}
    for item in fallback:
        if len(cleaned) >= 3:
            break
        name_norm = _norm(item.get("name", ""))
        if name_norm in existing_norm:
            continue
        cleaned.append(item)
        existing_norm.add(name_norm)
    return cleaned


def build_deterministic_decomposition(
    *,
    parsed: dict[str, Any],
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> DeterministicDecomposeResult:
    """Build deterministic nodes/edges from conceptual LLM output."""
    raw_subs = _prepare_raw_sub_nodes(parsed.get("sub_nodes"), parent=parent)

    nodes: list[AlgorithmicNode] = []
    for idx, raw in enumerate(raw_subs):
        if not isinstance(raw, dict):
            continue
        name = str(raw.get("name", "")).strip() or f"step_{idx + 1}"
        description = str(raw.get("description", "")).strip() or f"Sub-step {idx + 1}"
        concept_type = _infer_concept_type(raw, parent_type=parent.concept_type, catalog=catalog)
        inputs, outputs = _ensure_ports(raw, index=idx, total=len(raw_subs), parent=parent)

        prim_hint = str(
            raw.get("matched_primitive")
            or raw.get("matched_primitive_hint")
            or raw.get("atomic_hint")
            or ""
        ).strip()
        prim = _find_primitive(catalog, prim_hint) or _find_primitive(catalog, name)

        matched_primitive = prim.name if prim is not None else ""
        atomic_claim = bool(raw.get("is_atomic", False) or prim is not None)

        node = AlgorithmicNode(
            node_id=f"node_{uuid.uuid4().hex[:8]}",
            parent_id=parent.node_id,
            name=name,
            description=description,
            concept_type=concept_type,
            inputs=inputs,
            outputs=outputs,
            depth=parent.depth + 1,
            type_signature=str(raw.get("type_signature", "")).strip(),
            matched_primitive=matched_primitive or None,
            status=NodeStatus.PENDING,
        )

        is_atomic = atomic_claim and catalog.is_atomic(node)
        if is_atomic:
            node = node.model_copy(update={"status": NodeStatus.ATOMIC})
            if prim is not None and not node.type_signature:
                node = node.model_copy(update={"type_signature": prim.type_signature})

        if not node.type_signature:
            node = node.model_copy(update={"type_signature": _signature_from_ports(node.inputs, node.outputs)})

        nodes.append(node)

    edge_hints = _collect_edge_hints(parsed)
    edges = _sanitize_edge_hints(nodes, edge_hints)
    if not edges:
        edges = _build_chain_edges(nodes)

    return DeterministicDecomposeResult(nodes=nodes, edges=edges)

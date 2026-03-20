"""Deterministic post-processing for Architect decomposition payloads.

The LLM proposes conceptual sub-nodes (names/descriptions and optional hints).
This module deterministically synthesizes operational details:
- concept type fallback/inference,
- IO specs,
- atomic status and primitive binding,
- edge wiring and type propagation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
import re
import uuid
from typing import Any

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.architect.skeletons import get_skeleton, instantiate_skeleton
from sciona.architect.strategy_classifier import StrategyClassifier


@dataclass
class DeterministicDecomposeResult:
    """Deterministic decomposition artifacts."""

    nodes: list[AlgorithmicNode]
    edges: list[DependencyEdge]
    rewrite_actions: list[dict[str, Any]]


class DeterministicRewriteError(ValueError):
    """Raised when deterministic rewrite leaves an invalid typed subgraph."""


@dataclass(frozen=True)
class PrimitiveBindingSuggestion:
    """Deterministic primitive-binding candidate with confidence and provenance."""

    primitive: Any | None
    confidence: float
    source: str


@dataclass(frozen=True)
class ParsedDecomposePrompt:
    """Structured fields extracted from the Architect decompose prompt."""

    node_name: str
    node_description: str
    concept_type: ConceptType | None
    inputs: list[IOSpec]
    outputs: list[IOSpec]
    depth: int
    max_depth: int


_SUPPORTED_DETERMINISTIC_DECOMPOSE: tuple[ConceptType, ...] = (
    ConceptType.DIVIDE_AND_CONQUER,
    ConceptType.DYNAMIC_PROGRAMMING,
    ConceptType.SIGNAL_FILTER,
)


def _invariant_error(code: str, message: str) -> DeterministicRewriteError:
    return DeterministicRewriteError(f"[{code}] {message}")


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
    ConceptType.SIGNAL_TRANSFORM: [
        {
            "name": "Apply Window Function",
            "description": "Apply a deterministic window to the input signal segment.",
            "concept_type": ConceptType.SIGNAL_TRANSFORM.value,
            "matched_primitive_hint": "apply_window_function",
        },
        {
            "name": "Compute Forward Transform",
            "description": "Project the windowed signal into the transform domain.",
            "concept_type": ConceptType.SIGNAL_TRANSFORM.value,
            "matched_primitive_hint": "compute_forward_transform",
        },
        {
            "name": "Process Spectrum",
            "description": "Modify spectral coefficients to extract or suppress target content.",
            "concept_type": ConceptType.SIGNAL_TRANSFORM.value,
            "matched_primitive_hint": "process_spectrum",
        },
        {
            "name": "Compute Inverse Transform",
            "description": "Recover a time-domain signal from the modified spectrum.",
            "concept_type": ConceptType.SIGNAL_TRANSFORM.value,
            "matched_primitive_hint": "compute_inverse_transform",
        },
    ],
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

_PARENT_CONCEPTUAL_FALLBACKS: dict[str, list[dict[str, str]]] = {
    "design_filter": [dict(item) for item in _CONCEPTUAL_FALLBACKS[ConceptType.SIGNAL_FILTER]],
    "validate_stability": [
        {
            "name": "Normalize Coefficient Form",
            "description": "Normalize coefficient ordering and representation before stability analysis.",
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "canonicalize_filter_coefficients",
        },
        {
            "name": "Construct Characteristic Polynomial",
            "description": "Construct the characteristic polynomial associated with the normalized coefficients.",
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "construct_characteristic_polynomial",
        },
        {
            "name": "Compute Pole Locations",
            "description": "Compute discrete-time poles from the characteristic polynomial.",
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "compute_pole_locations",
        },
        {
            "name": "Evaluate Discrete-Time Stability",
            "description": "Assess whether the pole set satisfies the discrete-time stability criterion.",
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "assess_discrete_time_stability",
        },
        {
            "name": "Emit Stable Coefficients",
            "description": "Return validated coefficients once the stability report passes acceptance checks.",
            "concept_type": ConceptType.SIGNAL_FILTER.value,
            "matched_primitive_hint": "finalize_stable_coefficients",
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

_VALIDATION_WRAPPER_NAME_TOKENS = {
    "validate",
    "validation",
    "verify",
    "verification",
    "check",
    "checking",
    "finalize",
    "finalization",
}

_VALIDATION_EXEMPT_TOKENS = {
    "classify",
    "count",
    "decide",
    "detect",
    "estimate",
    "measure",
    "predicate",
    "score",
}

_VALIDATION_COMPUTE_TOKENS = {
    "compute",
    "derive",
    "generate",
    "synthesize",
    "construct",
    "apply",
    "assemble",
    "map",
    "translate",
    "select",
    "choose",
    "inspect",
    "assess",
}

_SUBSTANTIVE_COMPUTE_TOKENS = set(_VALIDATION_COMPUTE_TOKENS) | {"optimize", "fit"}

_PRIMITIVE_MATCH_STOPWORDS = {
    "a",
    "an",
    "and",
    "the",
    "to",
    "for",
    "from",
    "of",
    "with",
    "into",
    "on",
    "in",
    "by",
    "then",
    "after",
    "before",
    "result",
    "results",
    "step",
    "steps",
}

_ROUTING_WRAPPER_TOKENS = {
    "route",
    "routing",
    "pass",
    "handoff",
    "forward",
    "dispatch",
    "fallback",
    "branch",
    "branching",
    "escalate",
    "escalation",
    "retry",
    "failover",
}

_HELPER_PORT_TOKENS = {
    "config",
    "configuration",
    "context",
    "defaults",
    "grid",
    "options",
    "option",
    "parameters",
    "parameter",
    "plan",
    "policy",
    "schedule",
    "seed",
    "state",
}

_ATOMIC_BINDING_CONFIDENCE_THRESHOLD = 0.70

_SPECIALIZED_SCAFFOLD_PARENTS: dict[ConceptType, set[str]] = {
    ConceptType.SIGNAL_FILTER: {"design_filter"},
    ConceptType.SIGNAL_TRANSFORM: {"signal_transform"},
}


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _token_set(*values: str) -> set[str]:
    tokens: set[str] = set()
    for value in values:
        tokens.update(re.findall(r"[a-z0-9]+", value.lower()))
    return tokens


def _is_validation_wrapper(spec: dict[str, Any]) -> bool:
    name = str(spec.get("name", ""))
    description = str(spec.get("description", ""))
    tokens = _token_set(name, description)
    if not tokens & _VALIDATION_WRAPPER_NAME_TOKENS:
        return False
    if tokens & _VALIDATION_EXEMPT_TOKENS:
        return False
    if tokens & _VALIDATION_COMPUTE_TOKENS:
        return False
    return True


def _rewrite_validation_wrappers(raw_subs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    substantive = [item for item in raw_subs if not _is_validation_wrapper(item)]
    if len(substantive) >= 2:
        return substantive
    return raw_subs


def _primitive_match_tokens(*values: str) -> set[str]:
    tokens = _token_set(*values)
    return {token for token in tokens if token not in _PRIMITIVE_MATCH_STOPWORDS}


def _primitive_signature_text(primitive: Any) -> str:
    parts = [primitive.name, primitive.description]
    parts.extend(port.name for port in primitive.inputs)
    parts.extend(port.type_desc for port in primitive.inputs)
    parts.extend(port.name for port in primitive.outputs)
    parts.extend(port.type_desc for port in primitive.outputs)
    return " ".join(part for part in parts if part)


def _suggest_primitive_for_spec(
    spec: dict[str, Any],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> PrimitiveBindingSuggestion:
    spec_tokens = _primitive_match_tokens(
        str(spec.get("name", "")).strip(),
        str(spec.get("description", "")).strip(),
    )
    if spec_tokens & _ROUTING_WRAPPER_TOKENS and not spec_tokens & _SUBSTANTIVE_COMPUTE_TOKENS:
        return PrimitiveBindingSuggestion(None, 0.0, "")

    explicit = str(
        spec.get("matched_primitive")
        or spec.get("matched_primitive_hint")
        or spec.get("atomic_hint")
        or ""
    ).strip()
    if explicit:
        prim = _find_primitive(catalog, explicit)
        if prim is not None:
            return PrimitiveBindingSuggestion(prim, 1.0, "explicit_hint")

    name = str(spec.get("name", "")).strip()
    description = str(spec.get("description", "")).strip()
    for probe in (name, description, f"{name} {description}".strip()):
        if not probe:
            continue
        prim = _find_primitive(catalog, probe)
        if prim is not None:
            source = "exact_name" if probe == name else "exact_description"
            confidence = 0.95 if probe == name else 0.90
            return PrimitiveBindingSuggestion(prim, confidence, source)

    if not spec_tokens:
        return PrimitiveBindingSuggestion(None, 0.0, "")

    best_primitive = None
    best_score = 0.0
    for primitive in catalog.all_primitives():
        primitive_tokens = _primitive_match_tokens(_primitive_signature_text(primitive))
        if not primitive_tokens:
            continue
        overlap = spec_tokens & primitive_tokens
        if not overlap:
            continue

        score = float(len(overlap))
        if primitive.category == parent.concept_type:
            score += 1.5
        if _norm(primitive.name) in _norm(name):
            score += 1.0
        if len(overlap) >= max(2, len(spec_tokens) // 2):
            score += 0.5

        if score > best_score:
            best_score = score
            best_primitive = primitive

    if best_score >= 3.0:
        normalized = min(0.85, 0.30 + best_score * 0.10)
        return PrimitiveBindingSuggestion(best_primitive, normalized, "token_overlap")
    return PrimitiveBindingSuggestion(None, 0.0, "")


def _rewrite_conceptual_primitive_names(
    raw_subs: list[dict[str, Any]],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    for item in raw_subs:
        updated = dict(item)
        suggestion = _suggest_primitive_for_spec(updated, parent=parent, catalog=catalog)
        if suggestion.primitive is not None:
            updated.setdefault("matched_primitive_hint", suggestion.primitive.name)
            updated.setdefault("primitive_binding_confidence", suggestion.confidence)
            updated.setdefault("primitive_binding_source", suggestion.source)
        rewritten.append(updated)
    return rewritten


def _is_routing_wrapper(spec: dict[str, Any]) -> bool:
    if spec.get("matched_primitive") or spec.get("matched_primitive_hint"):
        return False
    tokens = _token_set(str(spec.get("name", "")), str(spec.get("description", "")))
    if not tokens & _ROUTING_WRAPPER_TOKENS:
        return False
    if tokens & _SUBSTANTIVE_COMPUTE_TOKENS:
        return False
    return True


def _rewrite_routing_wrappers(raw_subs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    substantive = [item for item in raw_subs if not _is_routing_wrapper(item)]
    if len(substantive) >= 2:
        return substantive
    return raw_subs


def _primitive_port_snapshot(spec: dict[str, Any], primitive: Any) -> tuple[list[IOSpec], list[IOSpec]]:
    inputs = _safe_iospec_list(spec.get("inputs"))
    outputs = _safe_iospec_list(spec.get("outputs"))
    if not inputs:
        inputs = _clone_ports(primitive.inputs)
    if not outputs:
        outputs = _clone_ports(primitive.outputs)
    return inputs, outputs


def _collapse_redundant_primitive_steps(
    raw_subs: list[dict[str, Any]],
    *,
    catalog: PrimitiveCatalog,
) -> list[dict[str, Any]]:
    best_by_primitive: dict[str, tuple[int, int]] = {}
    kept: list[dict[str, Any]] = []
    for item in raw_subs:
        primitive_name = str(
            item.get("matched_primitive")
            or item.get("matched_primitive_hint")
            or ""
        ).strip()
        if not primitive_name:
            kept.append(item)
            continue

        primitive = _find_primitive(catalog, primitive_name)
        if primitive is None:
            kept.append(item)
            continue

        specificity = len(_token_set(str(item.get("name", "")), str(item.get("description", ""))))
        specificity += len(_safe_iospec_list(item.get("inputs")))
        specificity += len(_safe_iospec_list(item.get("outputs")))

        existing = best_by_primitive.get(primitive.name)
        if existing is None:
            best_by_primitive[primitive.name] = (len(kept), specificity)
            kept.append(item)
            continue

        existing_index, existing_specificity = existing
        if specificity > existing_specificity:
            kept[existing_index] = item
            best_by_primitive[primitive.name] = (existing_index, specificity)

    return kept


def _port_is_available(port: IOSpec, available_ports: list[IOSpec]) -> bool:
    for candidate in available_ports:
        same_name = candidate.name == port.name
        compatible_type = (
            candidate.type_desc == port.type_desc
            or candidate.type_desc in {"", "Any"}
            or port.type_desc in {"", "Any"}
        )
        if same_name and compatible_type:
            return True
    return False


def _is_helper_like_port(port: IOSpec) -> bool:
    tokens = _token_set(port.name, port.type_desc, port.constraints)
    return bool(tokens & _HELPER_PORT_TOKENS)


def _synthesize_missing_input_helpers(
    raw_subs: list[dict[str, Any]],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> list[dict[str, Any]]:
    rewritten: list[dict[str, Any]] = []
    available_ports = _clone_ports(parent.inputs)

    for item in raw_subs:
        primitive_name = str(
            item.get("matched_primitive")
            or item.get("matched_primitive_hint")
            or ""
        ).strip()
        primitive = _find_primitive(catalog, primitive_name) if primitive_name else None
        if primitive is not None:
            for required_input in primitive.inputs:
                if not required_input.required:
                    continue
                if not _is_helper_like_port(required_input):
                    continue
                if _port_is_available(required_input, available_ports):
                    continue

                helper_name = f"Prepare {required_input.name.replace('_', ' ').title()}"
                if not any(_norm(existing.get("name", "")) == _norm(helper_name) for existing in rewritten):
                    rewritten.append(
                        {
                            "name": helper_name,
                            "description": (
                                f"Derive or configure {required_input.name} needed by "
                                f"{primitive.name}."
                            ),
                            "concept_type": ConceptType.DATA_ASSEMBLY.value,
                            "outputs": [
                                {
                                    "name": required_input.name,
                                    "type_desc": required_input.type_desc,
                                    "constraints": required_input.constraints,
                                    "required": required_input.required,
                                    "default_value_repr": required_input.default_value_repr,
                                }
                            ],
                        }
                    )
                    available_ports.append(required_input)

        rewritten.append(item)
        explicit_outputs = _safe_iospec_list(item.get("outputs"))
        if explicit_outputs:
            available_ports.extend(explicit_outputs)
        elif primitive is not None:
            available_ports.extend(_clone_ports(primitive.outputs))

    return rewritten


def _rewrite_raw_sub_nodes(
    raw_subs: list[dict[str, Any]],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rewrite_actions: list[dict[str, Any]] = []

    def _snapshot(items: list[dict[str, Any]]) -> list[tuple[str, str]]:
        return [
            (
                str(item.get("name", "")).strip(),
                str(
                    item.get("matched_primitive")
                    or item.get("matched_primitive_hint")
                    or ""
                ).strip(),
            )
            for item in items
            if isinstance(item, dict)
        ]

    def _record(stage: str, before: list[dict[str, Any]], after: list[dict[str, Any]]) -> None:
        before_snapshot = _snapshot(before)
        after_snapshot = _snapshot(after)
        removed = [name for name, _hint in before_snapshot if (name, _hint) not in after_snapshot]
        added = [name for name, _hint in after_snapshot if (name, _hint) not in before_snapshot]
        normalized: list[str] = []
        before_by_name = {name: hint for name, hint in before_snapshot if name}
        for name, hint in after_snapshot:
            if name in before_by_name and before_by_name[name] != hint and hint:
                normalized.append(name)
        if removed or added or normalized:
            rewrite_actions.append(
                {
                    "stage": stage,
                    "removed": removed,
                    "added": added,
                    "normalized": normalized,
                }
            )

    rewritten = _rewrite_validation_wrappers(raw_subs)
    _record("validation_wrapper_elision", raw_subs, rewritten)
    previous = rewritten
    rewritten = _rewrite_conceptual_primitive_names(
        previous,
        parent=parent,
        catalog=catalog,
    )
    _record("primitive_normalization", previous, rewritten)
    previous = rewritten
    rewritten = _rewrite_routing_wrappers(previous)
    _record("routing_wrapper_elision", previous, rewritten)
    previous = rewritten
    rewritten = _collapse_redundant_primitive_steps(previous, catalog=catalog)
    _record("redundant_primitive_collapse", previous, rewritten)
    previous = rewritten
    rewritten = _synthesize_missing_input_helpers(
        previous,
        parent=parent,
        catalog=catalog,
    )
    _record("helper_synthesis", previous, rewritten)
    previous = rewritten
    rewritten = _merge_specialized_fallbacks(
        previous,
        parent=parent,
        catalog=catalog,
    )
    _record("specialized_fallback_merge", previous, rewritten)
    return rewritten, rewrite_actions


def _canonical_primitive_name_for_spec(
    spec: dict[str, Any],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> str:
    explicit = str(
        spec.get("matched_primitive")
        or spec.get("matched_primitive_hint")
        or spec.get("atomic_hint")
        or ""
    ).strip()
    if explicit:
        prim = _find_primitive(catalog, explicit)
        if prim is not None:
            return prim.name
    suggestion = _suggest_primitive_for_spec(spec, parent=parent, catalog=catalog)
    return suggestion.primitive.name if suggestion.primitive is not None else ""


def _specialized_fallbacks_for_parent(
    parent: AlgorithmicNode,
) -> list[dict[str, str]] | None:
    parent_name = _norm(parent.name)
    parent_specific = _PARENT_CONCEPTUAL_FALLBACKS.get(parent_name)
    if parent_specific is not None:
        return [dict(item) for item in parent_specific]

    specialized = _CONCEPTUAL_FALLBACKS.get(parent.concept_type)
    if not specialized:
        return None
    allowed_parent_names = _SPECIALIZED_SCAFFOLD_PARENTS.get(parent.concept_type)
    if allowed_parent_names and parent_name not in allowed_parent_names:
        return None
    return [dict(item) for item in specialized]


def _merge_specialized_fallbacks(
    raw_subs: list[dict[str, Any]],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> list[dict[str, Any]]:
    specialized = _specialized_fallbacks_for_parent(parent)
    if not specialized:
        return raw_subs
    fallback_primitive_names = [
        str(item.get("matched_primitive_hint", "")).strip()
        for item in specialized
        if str(item.get("matched_primitive_hint", "")).strip()
    ]
    if fallback_primitive_names and not all(
        _find_primitive(catalog, primitive_name) is not None
        for primitive_name in fallback_primitive_names
    ):
        return raw_subs

    used_indices: set[int] = set()
    merged: list[dict[str, Any]] = []
    existing_primitive_names = [
        _canonical_primitive_name_for_spec(item, parent=parent, catalog=catalog)
        for item in raw_subs
    ]

    for fallback in specialized:
        fallback_prim = str(fallback.get("matched_primitive_hint", "")).strip()
        match_index = None
        for index, primitive_name in enumerate(existing_primitive_names):
            if index in used_indices:
                continue
            if fallback_prim and primitive_name == fallback_prim:
                match_index = index
                break
            if _norm(str(raw_subs[index].get("name", ""))) == _norm(str(fallback.get("name", ""))):
                match_index = index
                break

        if match_index is not None:
            merged.append(raw_subs[match_index])
            used_indices.add(match_index)
        else:
            merged.append(dict(fallback))

    return merged


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
        rows.append(
            IOSpec(
                name=name,
                type_desc=type_desc,
                constraints=constraints,
                required=bool(item.get("required", True)),
                default_value_repr=str(item.get("default_value_repr", "") or ""),
            )
        )
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
        return [
            IOSpec(
                name=i.name,
                type_desc=i.type_desc,
                constraints=i.constraints,
                required=i.required,
                default_value_repr=i.default_value_repr,
            )
            for i in parent.inputs
        ]
    return [IOSpec(name="input", type_desc="Any")]


def _default_outputs(parent: AlgorithmicNode) -> list[IOSpec]:
    if parent.outputs:
        return [
            IOSpec(
                name=o.name,
                type_desc=o.type_desc,
                constraints=o.constraints,
                required=o.required,
                default_value_repr=o.default_value_repr,
            )
            for o in parent.outputs
        ]
    return [IOSpec(name="result", type_desc="Any")]


def _clone_ports(ports: list[IOSpec]) -> list[IOSpec]:
    return [
        IOSpec(
            name=port.name,
            type_desc=port.type_desc,
            constraints=port.constraints,
            required=port.required,
            default_value_repr=port.default_value_repr,
        )
        for port in ports
    ]


def _clone_port(port: IOSpec) -> IOSpec:
    return IOSpec(
        name=port.name,
        type_desc=port.type_desc,
        constraints=port.constraints,
        required=port.required,
        default_value_repr=port.default_value_repr,
    )


def _port(
    name: str,
    type_desc: str,
    constraints: str = "",
    *,
    required: bool = True,
    default_value_repr: str = "",
) -> IOSpec:
    return IOSpec(
        name=name,
        type_desc=type_desc,
        constraints=constraints,
        required=required,
        default_value_repr=default_value_repr,
    )


def _first_type(ports: list[IOSpec], default: str) -> str:
    for port in ports:
        if port.type_desc.strip():
            return port.type_desc
    return default


def _signal_filter_ports(
    parent: AlgorithmicNode,
    spec: dict[str, Any],
    *,
    index: int,
    total: int,
) -> tuple[list[IOSpec], list[IOSpec]] | None:
    parent_name = _norm(parent.name)
    step_name = _norm(str(spec.get("name", "")))
    parent_inputs = _clone_ports(parent.inputs)
    parent_outputs = _clone_ports(parent.outputs)
    input_type = _first_type(parent_inputs, "np.ndarray")
    output_type = _first_type(parent_outputs, "np.ndarray")

    if parent_name == "design_filter":
        if "parse" in step_name or "normalize" in step_name or "interpret" in step_name:
            return (
                parent_inputs or [_port("spec", "filter specification")],
                [_port("design_targets", "filter design targets")],
            )
        if "target" in step_name or "constraint" in step_name:
            return (
                parent_inputs or [_port("spec", "filter specification")],
                [_port("design_targets", "filter design targets")],
            )
        if "select" in step_name or "choose" in step_name:
            return (
                [_port("design_targets", "filter design targets")],
                [_port("design_strategy", "filter design strategy")],
            )
        if "synth" in step_name or "coeff" in step_name:
            return (
                [_port("design_strategy", "filter design strategy")],
                [_port("candidate_coefficients", "filter coefficients")],
            )
        if "evaluate" in step_name or "refine" in step_name or "final" in step_name:
            return (
                [
                    _port("candidate_coefficients", "filter coefficients"),
                    _port("design_targets", "filter design targets"),
                ],
                parent_outputs or [_port("coefficients", "filter coefficients")],
            )

    if parent_name == "validate_stability":
        if "normalize" in step_name or "canonical" in step_name:
            return (
                parent_inputs or [_port("coefficients", "filter coefficients")],
                [_port("normalized_coefficients", "filter coefficients")],
            )
        if "character" in step_name or "dynamics" in step_name or "construct" in step_name:
            return (
                [_port("normalized_coefficients", "filter coefficients")],
                [_port("characteristic_polynomial", "np.polynomial.Polynomial")],
            )
        if "pole" in step_name or "solve" in step_name:
            return (
                [_port("characteristic_polynomial", "np.polynomial.Polynomial")],
                [_port("poles", "np.ndarray")],
            )
        if "margin" in step_name or "stability" in step_name:
            return (
                [_port("poles", "np.ndarray")],
                [_port("stability_report", "stability report")],
            )
        if "emit" in step_name or "pass" in step_name or "final" in step_name:
            return (
                [
                    _port("normalized_coefficients", "filter coefficients"),
                    _port("stability_report", "stability report"),
                ],
                parent_outputs or [_port("valid_coefficients", "filter coefficients")],
            )

    if parent_name == "apply_filter":
        if "validate" in step_name or "precondition" in step_name:
            return (
                parent_inputs
                or [
                    _port("valid_coefficients", "filter coefficients"),
                    _port("signal", input_type),
                ],
                [
                    _port("validated_coefficients", "filter coefficients"),
                    _port("validated_signal", input_type),
                ],
            )
        if "boundary" in step_name or "initial" in step_name or "policy" in step_name:
            return (
                [
                    _port("validated_coefficients", "filter coefficients"),
                    _port("validated_signal", input_type),
                ],
                [_port("filter_plan", "filter execution plan")],
            )
        if "apply" in step_name or "execute" in step_name:
            return (
                [_port("filter_plan", "filter execution plan")],
                [_port("filtered_signal", output_type)],
            )
        if "transient" in step_name or "edge" in step_name or "mitigate" in step_name:
            return (
                [_port("filtered_signal", output_type)],
                [_port("stabilized_signal", output_type)],
            )
        if "final" in step_name:
            return (
                [_port("stabilized_signal", output_type)],
                parent_outputs or [_port("filtered", output_type)],
            )

    if parent_name == "frequency_response":
        response_type = _first_type(parent_outputs, "tuple[np.ndarray, np.ndarray]")
        if "standard" in step_name or "canonical" in step_name or "normalize" in step_name:
            return (
                parent_inputs or [_port("valid_coefficients", "filter coefficients")],
                [_port("normalized_coefficients", "filter coefficients")],
            )
        if "grid" in step_name or "domain" in step_name or "sampling" in step_name:
            return (
                [_port("normalized_coefficients", "filter coefficients")],
                [_port("frequency_grid", "np.ndarray")],
            )
        if "complex" in step_name or "transfer" in step_name or "compute" in step_name:
            return (
                [
                    _port("normalized_coefficients", "filter coefficients"),
                    _port("frequency_grid", "np.ndarray"),
                ],
                [_port("complex_response", "np.ndarray")],
            )
        if "extract" in step_name or "derive" in step_name or "view" in step_name:
            return (
                [_port("complex_response", "np.ndarray")],
                [_port("response_summary", response_type)],
            )
        if "assess" in step_name or "inspect" in step_name:
            return (
                [_port("response_summary", response_type)],
                [_port("band_assessment", "response assessment")],
            )
        if "assemble" in step_name or "final" in step_name:
            return (
                [_port("response_summary", response_type)],
                parent_outputs or [_port("response", response_type)],
            )

    if parent_name in {
        "design_filter",
        "validate_stability",
        "apply_filter",
        "frequency_response",
    }:
        fallback_type = output_type if output_type != "Any" else input_type
        inputs = (
            parent_inputs
            if index == 0
            else [_port(f"step_{index}_input", fallback_type)]
        )
        outputs = (
            parent_outputs
            if index == total - 1
            else [_port(f"step_{index + 1}_artifact", fallback_type)]
        )
        return (inputs, outputs)

    return None


def _signal_transform_ports(
    parent: AlgorithmicNode,
    spec: dict[str, Any],
    *,
    index: int,
    total: int,
) -> tuple[list[IOSpec], list[IOSpec]] | None:
    parent_name = _norm(parent.name)
    step_name = _norm(str(spec.get("name", "")))
    parent_inputs = _clone_ports(parent.inputs)
    parent_outputs = _clone_ports(parent.outputs)
    signal_type = _first_type(parent_inputs, "np.ndarray")
    result_type = _first_type(parent_outputs, "np.ndarray")

    if parent_name == "window":
        if any(token in step_name for token in ("select", "config", "parameter", "prepare")):
            return (
                parent_inputs or [_port("signal", signal_type)],
                [_port("window_config", "window configuration")],
            )
        if any(token in step_name for token in ("kernel", "weights", "coeff")):
            return (
                [_port("window_config", "window configuration")],
                [_port("window_kernel", "np.ndarray")],
            )
        if any(token in step_name for token in ("apply", "multiply", "window")):
            return (
                [
                    _port("signal", signal_type),
                    _port("window_kernel", "np.ndarray"),
                ],
                parent_outputs or [_port("windowed", "np.ndarray")],
            )

    if parent_name == "forward_transform":
        if any(token in step_name for token in ("prepare", "plan", "pad", "layout")):
            return (
                parent_inputs or [_port("windowed", "np.ndarray")],
                [_port("transform_plan", "transform plan")],
            )
        if any(token in step_name for token in ("forward", "transform", "fft", "dct", "stft")):
            inputs = [_port("windowed", "np.ndarray")]
            if index > 0:
                inputs.append(_port("transform_plan", "transform plan", required=False))
            return (
                inputs,
                parent_outputs or [_port("spectrum", "np.ndarray")],
            )

    if parent_name == "spectral_processing":
        if any(token in step_name for token in ("prepare", "mask", "weights", "filter", "band")):
            return (
                parent_inputs or [_port("spectrum", "np.ndarray")],
                [_port("spectral_operator", "spectral operator")],
            )
        if any(token in step_name for token in ("apply", "process", "modify", "suppress", "enhance")):
            return (
                [
                    _port("spectrum", "np.ndarray"),
                    _port("spectral_operator", "spectral operator", required=False),
                ],
                parent_outputs or [_port("modified_spectrum", "np.ndarray")],
            )

    if parent_name == "inverse_transform":
        if any(token in step_name for token in ("prepare", "plan", "layout")):
            return (
                parent_inputs or [_port("modified_spectrum", "np.ndarray")],
                [_port("inverse_plan", "transform plan")],
            )
        if any(token in step_name for token in ("inverse", "ifft", "reconstruct", "recover", "synthesize")):
            inputs = [_port("modified_spectrum", "np.ndarray")]
            if index > 0:
                inputs.append(_port("inverse_plan", "transform plan", required=False))
            return (
                inputs,
                parent_outputs or [_port("result", result_type)],
            )

    if parent_name in {
        "window",
        "forward_transform",
        "spectral_processing",
        "inverse_transform",
    }:
        fallback_type = result_type if result_type != "Any" else signal_type
        inputs = parent_inputs if index == 0 else [_port(f"step_{index}_input", fallback_type)]
        outputs = (
            parent_outputs
            if index == total - 1
            else [_port(f"step_{index + 1}_artifact", fallback_type)]
        )
        return (inputs, outputs)

    return None


def _synthesize_specialized_ports(
    parent: AlgorithmicNode,
    spec: dict[str, Any],
    *,
    index: int,
    total: int,
) -> tuple[list[IOSpec], list[IOSpec]] | None:
    if parent.concept_type == ConceptType.SIGNAL_FILTER:
        return _signal_filter_ports(parent, spec, index=index, total=total)
    if parent.concept_type == ConceptType.SIGNAL_TRANSFORM:
        return _signal_transform_ports(parent, spec, index=index, total=total)
    return None


def _ensure_ports(
    spec: dict[str, Any],
    *,
    index: int,
    total: int,
    parent: AlgorithmicNode,
) -> tuple[list[IOSpec], list[IOSpec]]:
    inputs = _safe_iospec_list(spec.get("inputs"))
    outputs = _safe_iospec_list(spec.get("outputs"))

    specialized = _synthesize_specialized_ports(
        parent,
        spec,
        index=index,
        total=total,
    )
    if specialized is not None:
        if not inputs:
            inputs = specialized[0]
        if not outputs:
            outputs = specialized[1]

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


def _port_is_typed(port: IOSpec) -> bool:
    return port.type_desc.strip() not in {"", "Any"}


def _primary_type(parent: AlgorithmicNode) -> str:
    for port in parent.outputs + parent.inputs:
        if _port_is_typed(port):
            return port.type_desc
    return "Any"


def _repair_ports_from_reference(
    current: list[IOSpec],
    reference: list[IOSpec],
    *,
    fallback_type: str,
    allow_extension: bool = True,
    rename_generic_names: bool = True,
    force_reference_names: bool = False,
    force_reference_types: bool = False,
) -> list[IOSpec]:
    if not current and not reference:
        return []

    refs_by_name = {port.name: port for port in reference}
    repaired: list[IOSpec] = []
    limit = max(len(current), len(reference)) if allow_extension else len(current)

    for index in range(limit):
        base = current[index] if index < len(current) else None
        ref = None
        if base is not None:
            ref = refs_by_name.get(base.name)
        if ref is None and index < len(reference):
            ref = reference[index]

        if base is None and ref is not None:
            repaired.append(_clone_port(ref))
            continue
        if base is None:
            continue

        name = base.name
        if ref is not None and force_reference_names:
            name = ref.name
        elif rename_generic_names and ref is not None and name in {"", "input", "result", f"step_{index}_input", f"step_{index + 1}_artifact"}:
            name = ref.name

        type_desc = base.type_desc.strip() or "Any"
        if ref is not None and force_reference_types and _port_is_typed(ref):
            type_desc = ref.type_desc
        elif type_desc == "Any":
            if ref is not None and _port_is_typed(ref):
                type_desc = ref.type_desc
            elif fallback_type not in {"", "Any"}:
                type_desc = fallback_type

        constraints = base.constraints or (ref.constraints if ref is not None else "")
        repaired.append(
            IOSpec(
                name=name,
                type_desc=type_desc,
                constraints=constraints,
                required=base.required if base is not None else (ref.required if ref is not None else True),
                default_value_repr=(
                    base.default_value_repr
                    or (ref.default_value_repr if ref is not None else "")
                ),
            )
        )

    return repaired


def _repair_node_ports(
    nodes: list[AlgorithmicNode],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> list[AlgorithmicNode]:
    if not nodes:
        return []

    fallback_type = _primary_type(parent)
    repaired_nodes: list[AlgorithmicNode] = []

    # Primitive signatures are the strongest source of truth.
    for node in nodes:
        primitive = _find_primitive(catalog, node.matched_primitive or "")
        inputs = _clone_ports(node.inputs)
        outputs = _clone_ports(node.outputs)
        if primitive is not None:
            inputs = _repair_ports_from_reference(
                inputs,
                _clone_ports(primitive.inputs),
                fallback_type=fallback_type,
                force_reference_names=True,
                force_reference_types=True,
            )
            outputs = _repair_ports_from_reference(
                outputs,
                _clone_ports(primitive.outputs),
                fallback_type=fallback_type,
                force_reference_names=True,
                force_reference_types=True,
            )
        repaired_nodes.append(node.model_copy(update={"inputs": inputs, "outputs": outputs}))

    # Propagate typed ports through the chain.
    propagated: list[AlgorithmicNode] = []
    for index, node in enumerate(repaired_nodes):
        primitive_bound = _find_primitive(catalog, node.matched_primitive or "") is not None
        input_reference = parent.inputs if index == 0 else propagated[index - 1].outputs
        output_reference = (
            parent.outputs
            if index == len(repaired_nodes) - 1
            else repaired_nodes[index + 1].inputs
        )
        propagated.append(
            node.model_copy(
                update={
                    "inputs": _repair_ports_from_reference(
                        _clone_ports(node.inputs),
                        _clone_ports(input_reference),
                        fallback_type=fallback_type,
                        allow_extension=False,
                        rename_generic_names=not primitive_bound,
                    ),
                    "outputs": _repair_ports_from_reference(
                        _clone_ports(node.outputs),
                        _clone_ports(output_reference),
                        fallback_type=fallback_type,
                        allow_extension=False,
                        rename_generic_names=not primitive_bound,
                    ),
                }
            )
        )

    return propagated


def _validate_rewritten_nodes(
    nodes: list[AlgorithmicNode],
    *,
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
) -> None:
    parent_is_typed = any(_port_is_typed(port) for port in parent.inputs + parent.outputs)
    seen_names: set[str] = set()
    for node in nodes:
        normalized = _norm(node.name)
        if normalized in seen_names:
            raise _invariant_error(
                "duplicate_child_name",
                f"Duplicate child name '{node.name}' under parent '{parent.name}'",
            )
        seen_names.add(normalized)

    if not parent_is_typed:
        return
    for node in nodes:
        unresolved = [port.name for port in node.inputs + node.outputs if not _port_is_typed(port)]
        if unresolved:
            raise _invariant_error(
                "unresolved_any_ports",
                f"Unresolved Any ports remain for child '{node.name}' under typed parent "
                f"'{parent.name}': {', '.join(unresolved)}",
            )
        primitive = _find_primitive(catalog, node.matched_primitive or "")
        if primitive is None:
            continue
        allowed_inputs = {port.name for port in primitive.inputs}
        allowed_outputs = {port.name for port in primitive.outputs}
        extra_inputs = [port.name for port in node.inputs if port.name not in allowed_inputs]
        extra_outputs = [port.name for port in node.outputs if port.name not in allowed_outputs]
        if extra_inputs or extra_outputs:
            problems = []
            if extra_inputs:
                problems.append(f"extra inputs: {', '.join(extra_inputs)}")
            if extra_outputs:
                problems.append(f"extra outputs: {', '.join(extra_outputs)}")
            raise _invariant_error(
                "primitive_signature_violation",
                f"Primitive-bound child '{node.name}' violates primitive signature for "
                f"'{primitive.name}' under parent '{parent.name}': {'; '.join(problems)}",
            )


def _validate_rewritten_graph(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
    *,
    parent: AlgorithmicNode,
) -> None:
    if len(nodes) <= 1:
        return

    by_id = {node.node_id: node for node in nodes}
    incoming: dict[str, int] = {node.node_id: 0 for node in nodes}
    outgoing: dict[str, int] = {node.node_id: 0 for node in nodes}
    adjacency: dict[str, list[str]] = {node.node_id: [] for node in nodes}

    for edge in edges:
        if edge.source_id == edge.target_id:
            raise _invariant_error(
                "self_loop",
                f"Self-loop detected on child '{by_id.get(edge.source_id, parent).name}' under parent '{parent.name}'",
            )
        if edge.source_id not in by_id or edge.target_id not in by_id:
            continue
        outgoing[edge.source_id] += 1
        incoming[edge.target_id] += 1
        adjacency[edge.source_id].append(edge.target_id)

    for node in nodes:
        if incoming[node.node_id] == 0 and outgoing[node.node_id] == 0:
            raise _invariant_error(
                "disconnected_child",
                f"Child '{node.name}' is disconnected under parent '{parent.name}'",
            )

    visiting: set[str] = set()
    visited: set[str] = set()

    def _visit(node_id: str) -> None:
        if node_id in visited:
            return
        if node_id in visiting:
            raise _invariant_error(
                "cycle_detected",
                f"Cycle detected in rewritten child graph under parent '{parent.name}'",
            )
        visiting.add(node_id)
        for target_id in adjacency[node_id]:
            _visit(target_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for node_id in adjacency:
        _visit(node_id)

    parent_input_names = {port.name for port in parent.inputs}
    parent_output_names = {port.name for port in parent.outputs}
    roots = [
        node.node_id
        for node in nodes
        if incoming[node.node_id] == 0 and any(port.name in parent_input_names for port in node.inputs)
    ]
    sinks = {
        node.node_id
        for node in nodes
        if outgoing[node.node_id] == 0 and any(port.name in parent_output_names for port in node.outputs)
    }
    if roots and sinks:
        frontier = list(roots)
        seen = set(frontier)
        while frontier:
            current = frontier.pop()
            if current in sinks:
                return
            for target_id in adjacency[current]:
                if target_id not in seen:
                    seen.add(target_id)
                    frontier.append(target_id)
        raise _invariant_error(
            "missing_typed_path",
            f"No typed path connects parent inputs to outputs under '{parent.name}'",
        )


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
    node_order = {node.node_id: index for index, node in enumerate(nodes)}
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
        if node_order.get(src.node_id, -1) >= node_order.get(tgt.node_id, -1):
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


def _synthesize_matching_input_edges(
    nodes: list[AlgorithmicNode],
    existing_edges: list[DependencyEdge],
) -> list[DependencyEdge]:
    if len(nodes) <= 1:
        return existing_edges

    seen = {_edge_tuple(edge) for edge in existing_edges}
    nodes_by_id = {node.node_id: node for node in nodes}
    incoming_inputs: dict[str, set[str]] = {}
    for edge in existing_edges:
        target = nodes_by_id.get(edge.target_id)
        if target is None:
            continue
        target_input = next((port for port in target.inputs if port.name == edge.input_name), None)
        if target_input is None:
            continue
        if (
            target_input.type_desc in {"", "Any"}
            or edge.source_type == target_input.type_desc
            and edge.target_type == target_input.type_desc
        ):
            incoming_inputs.setdefault(edge.target_id, set()).add(edge.input_name)

    augmented = list(existing_edges)
    for target_index, target in enumerate(nodes):
        satisfied = incoming_inputs.setdefault(target.node_id, set())
        required_inputs = [port for port in target.inputs if port.required]
        if not required_inputs:
            continue

        for target_input in required_inputs:
            if target_input.name in satisfied:
                continue
            for source in reversed(nodes[:target_index]):
                match = next(
                    (
                        output
                        for output in source.outputs
                        if output.name == target_input.name
                        and output.type_desc == target_input.type_desc
                    ),
                    None,
                )
                if match is None:
                    continue
                edge = DependencyEdge(
                    source_id=source.node_id,
                    target_id=target.node_id,
                    output_name=match.name,
                    input_name=target_input.name,
                    source_type=match.type_desc,
                    target_type=target_input.type_desc,
                    requires_glue=False,
                )
                key = _edge_tuple(edge)
                if key in seen:
                    break
                augmented.append(edge)
                seen.add(key)
                satisfied.add(target_input.name)
                break

    return augmented


def _prune_conflicting_typed_edges(
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
) -> list[DependencyEdge]:
    if not edges:
        return edges

    nodes_by_id = {node.node_id: node for node in nodes}
    exact_typed_inputs: set[tuple[str, str]] = set()
    for edge in edges:
        target = nodes_by_id.get(edge.target_id)
        if target is None:
            continue
        target_input = next((port for port in target.inputs if port.name == edge.input_name), None)
        if target_input is None:
            continue
        if (
            target_input.type_desc not in {"", "Any"}
            and edge.source_type == target_input.type_desc
            and edge.target_type == target_input.type_desc
        ):
            exact_typed_inputs.add((edge.target_id, edge.input_name))

    pruned: list[DependencyEdge] = []
    for edge in edges:
        target = nodes_by_id.get(edge.target_id)
        target_input = None if target is None else next(
            (port for port in target.inputs if port.name == edge.input_name),
            None,
        )
        if (
            target_input is not None
            and (edge.target_id, edge.input_name) in exact_typed_inputs
            and target_input.type_desc not in {"", "Any"}
            and (edge.source_type != target_input.type_desc or edge.target_type != target_input.type_desc)
        ):
            continue
        pruned.append(edge)
    return pruned


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
    specialized = _specialized_fallbacks_for_parent(parent)
    if specialized:
        return specialized
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


def _extract_prompt_value(user: str, label: str) -> str:
    pattern = re.compile(rf"^\s*{re.escape(label)}:\s*(.*)$", re.MULTILINE)
    match = pattern.search(user)
    return match.group(1).strip() if match is not None else ""


def _parse_prompt_io(raw: str) -> list[IOSpec]:
    text = raw.strip()
    if not text or text.lower() == "none":
        return []
    parts = re.split(r", (?=[A-Za-z_][A-Za-z0-9_]*: )", text)
    ports: list[IOSpec] = []
    for part in parts:
        if ":" not in part:
            continue
        name, rest = part.split(":", 1)
        port_name = name.strip()
        if not port_name:
            continue
        required = "(optional" not in rest
        default_match = re.search(r"default=([^)]+)", rest)
        type_desc = re.sub(r"\s*\(optional.*\)$", "", rest).strip()
        ports.append(
            IOSpec(
                name=port_name,
                type_desc=type_desc or "Any",
                required=required,
                default_value_repr=default_match.group(1).strip() if default_match else "",
            )
        )
    return ports


def _parse_decompose_prompt(user: str) -> ParsedDecomposePrompt:
    concept_raw = _extract_prompt_value(user, "Concept type")
    try:
        concept_type = ConceptType(concept_raw)
    except ValueError:
        concept_type = None
    try:
        depth = int(_extract_prompt_value(user, "Current depth") or 0)
    except ValueError:
        depth = 0
    try:
        max_depth = int(_extract_prompt_value(user, "Max depth") or 0)
    except ValueError:
        max_depth = 0
    return ParsedDecomposePrompt(
        node_name=_extract_prompt_value(user, "Name"),
        node_description=_extract_prompt_value(user, "Description"),
        concept_type=concept_type,
        inputs=_parse_prompt_io(_extract_prompt_value(user, "Inputs")),
        outputs=_parse_prompt_io(_extract_prompt_value(user, "Outputs")),
        depth=depth,
        max_depth=max_depth,
    )


def _goal_text_from_prompt(prompt: ParsedDecomposePrompt) -> str:
    parts = [prompt.node_name, prompt.node_description]
    if prompt.concept_type is not None:
        parts.append(prompt.concept_type.value.replace("_", " "))
    return " ".join(part for part in parts if part).strip()


def _is_template_node_name(concept: ConceptType, node_name: str) -> bool:
    skeleton = get_skeleton(concept)
    if skeleton is None:
        return False
    target = _norm(node_name)
    return any(_norm(node.name) == target for node in skeleton.template_nodes)


def _choose_decomposition_strategy(
    prompt: ParsedDecomposePrompt,
    classifier: StrategyClassifier,
) -> tuple[ConceptType, float, str, str] | None:
    goal_text = _goal_text_from_prompt(prompt)
    decision = classifier.classify(goal_text, allowed=list(_SUPPORTED_DETERMINISTIC_DECOMPOSE))
    if decision is not None and decision[1] >= 0.8:
        return decision
    if (
        prompt.concept_type in _SUPPORTED_DETERMINISTIC_DECOMPOSE
        and not _is_template_node_name(prompt.concept_type, prompt.node_name)
    ):
        return (
            prompt.concept_type,
            0.85,
            "deterministic parent concept fallback",
            "",
        )
    return None


def _emit_from_skeleton(
    strategy: ConceptType,
    *,
    prompt: ParsedDecomposePrompt,
    variant_hint: str = "",
) -> dict[str, Any] | None:
    skeleton = get_skeleton(strategy, variant=variant_hint or None)
    if skeleton is None:
        return None
    goal_text = _goal_text_from_prompt(prompt) or prompt.node_name or strategy.value.replace("_", " ")
    nodes, edges = instantiate_skeleton(skeleton, goal_text, parent_id="deterministic_parent", base_depth=0)
    if len(nodes) < 2:
        return None

    sub_nodes: list[dict[str, Any]] = []
    for idx, node in enumerate(nodes):
        inputs = prompt.inputs if idx == 0 and prompt.inputs else node.inputs
        outputs = prompt.outputs if idx == len(nodes) - 1 and prompt.outputs else node.outputs
        sub_nodes.append(
            {
                "name": node.name,
                "description": node.description,
                "concept_type": node.concept_type.value,
                "inputs": [port.model_dump() for port in inputs],
                "outputs": [port.model_dump() for port in outputs],
            }
        )

    node_names = {node.node_id: node.name for node in nodes}
    flow_hints = [
        {
            "from": node_names[edge.source_id],
            "to": node_names[edge.target_id],
            "why": f"{edge.output_name} feeds {edge.input_name}",
        }
        for edge in edges
        if edge.source_id in node_names and edge.target_id in node_names
    ]
    return {
        "progress_updates": [
            f"instantiate {strategy.value} decomposition skeleton",
            f"emit {len(sub_nodes)} conceptual sub-steps",
        ],
        "sub_nodes": sub_nodes,
        "flow_hints": flow_hints,
    }


class DeterministicDecomposer:
    """Deterministic skeleton-backed wrapper for architect_decompose."""

    _telemetry_provider = "deterministic"
    _telemetry_model = "decomposer_v1"

    def __init__(self, fallback: Any) -> None:
        self._fallback = fallback
        self._strategy_classifier = StrategyClassifier(fallback)
        self._last_completion_metadata: dict[str, Any] = {}
        self._last_error_metadata: dict[str, Any] = {}

    def get_last_completion_metadata(self) -> dict[str, Any]:
        return dict(self._last_completion_metadata)

    def get_last_error_metadata(self) -> dict[str, Any]:
        return dict(self._last_error_metadata)

    async def complete(self, system: str, user: str) -> str:
        prompt = _parse_decompose_prompt(user)
        decision = _choose_decomposition_strategy(prompt, self._strategy_classifier)
        if decision is None:
            self._last_completion_metadata = {"decompose_source": "fallback"}
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        strategy, confidence, rationale, variant_hint = decision
        payload = _emit_from_skeleton(
            strategy,
            prompt=prompt,
            variant_hint=variant_hint,
        )
        if payload is None:
            self._last_completion_metadata = {"decompose_source": "fallback"}
            self._last_error_metadata = {}
            return await self._fallback.complete(system, user)

        self._last_completion_metadata = {
            "decompose_source": "deterministic",
            "decompose_strategy": strategy.value,
            "decompose_confidence": round(confidence, 3),
            "decompose_variant_hint": variant_hint,
            "decompose_rationale": rationale,
        }
        self._last_error_metadata = {}
        return json.dumps(payload)

    async def complete_with_grammar(self, system: str, user: str, grammar: str) -> str:
        return await self.complete(system, user)


def build_deterministic_decomposition(
    *,
    parsed: dict[str, Any],
    parent: AlgorithmicNode,
    catalog: PrimitiveCatalog,
    use_monadic_rewriter: bool = False,
) -> DeterministicDecomposeResult:
    """Build deterministic nodes/edges from conceptual LLM output."""
    if use_monadic_rewriter:
        # -------------------------------------------------------------------
        # Formal Monadic Graph Rewriting (DPO Logic)
        # -------------------------------------------------------------------
        from sciona.architect.graph_rewriter import GraphRewriter, PriorityStrategy
        
        # In a full implementation, we would construct a CDG from 'parsed'
        # and apply a sequence of RewriteRules using the GraphState monad.
        # For this refactor, we maintain the interface but flag the gate.
        pass

    raw_subs = _prepare_raw_sub_nodes(parsed.get("sub_nodes"), parent=parent)
    raw_subs, rewrite_actions = _rewrite_raw_sub_nodes(raw_subs, parent=parent, catalog=catalog)
    raw_subs = _prepare_raw_sub_nodes(raw_subs, parent=parent)

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
        binding_confidence = float(raw.get("primitive_binding_confidence", 0.0) or 0.0)
        binding_source = str(raw.get("primitive_binding_source", "")).strip()
        prim = None
        if prim_hint:
            prim = _find_primitive(catalog, prim_hint)
            if prim is not None and binding_confidence <= 0.0:
                binding_confidence = 1.0 if binding_source == "explicit_hint" else 0.95
                binding_source = binding_source or "explicit_hint"
        if prim is None:
            suggestion = _suggest_primitive_for_spec(raw, parent=parent, catalog=catalog)
            prim = suggestion.primitive
            if suggestion.primitive is not None and suggestion.confidence > binding_confidence:
                binding_confidence = suggestion.confidence
                binding_source = suggestion.source
        if prim is None:
            prim = _find_primitive(catalog, name)
            if prim is not None and binding_confidence <= 0.0:
                binding_confidence = 0.95
                binding_source = "exact_name"

        matched_primitive = prim.name if prim is not None else ""
        strong_binding = prim is not None and binding_confidence >= _ATOMIC_BINDING_CONFIDENCE_THRESHOLD
        atomic_claim = bool(raw.get("is_atomic", False) or strong_binding)

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
            primitive_binding_confidence=binding_confidence,
            primitive_binding_source=binding_source,
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

    nodes = _repair_node_ports(nodes, parent=parent, catalog=catalog)
    _validate_rewritten_nodes(nodes, parent=parent, catalog=catalog)

    edge_hints = _collect_edge_hints(parsed)
    edges = _sanitize_edge_hints(nodes, edge_hints)
    if not edges:
        edges = _build_chain_edges(nodes)
    edges = _synthesize_matching_input_edges(nodes, edges)
    edges = _prune_conflicting_typed_edges(nodes, edges)
    _validate_rewritten_graph(nodes, edges, parent=parent)

    return DeterministicDecomposeResult(
        nodes=nodes,
        edges=edges,
        rewrite_actions=rewrite_actions,
    )

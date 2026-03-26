"""Phase 3: Deterministic code generation from a ValidatedMacroPlan.

Generates Pydantic state models, atom wrappers with ``@register_atom``
and ``icontract`` decorators, ghost witness functions, CDGExport nodes
and edges, and pre-filled MatchResults.
"""

from __future__ import annotations

import asyncio
import ast
import json
import re
from pathlib import Path

from sciona.json_utils import extract_json
import logging

from sciona.hunter.llm import LLMClient
from sciona.ingester.ffi_emitter import generate_ffi_bindings, generate_ffi_imports
from sciona.llm_router import INGESTER_OPAQUE_WITNESS, select_llm
from sciona.shared_context import (
    SharedContextMetrics,
    SharedContextStore,
    format_context_block,
)

from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.ingester.models import (
    ConceptualProfile,
    IngestIRPlan,
    IngestionBundle,
    MacroAtomSpec,
    MethodBinding,
    OutputBindingSpec,
    PlannedOperationGroup,
    ProposedMacroPlan,
    RawDataFlowGraph,
    StateEffectSpec,
    StateModelSpec,
    StateSlotSpec,
    OperationSpec,
    ValidatedMacroPlan,
)
from sciona.ingester.prompts import (
    DRAFT_OPAQUE_WITNESS_SYSTEM,
    DRAFT_OPAQUE_WITNESS_USER,
)
from sciona.types import (
    CandidateMatch,
    Declaration,
    MatchResult,
    PDGNode,
    Prover,
    VerificationLevel,
    VerificationResult,
)

logger = logging.getLogger(__name__)

# Bayesian concept types that get specialized witness templates
_BAYESIAN_CONCEPT_TYPES = frozenset(
    {
        ConceptType.SAMPLER,
        ConceptType.LOG_PROB,
        ConceptType.POSTERIOR_UPDATE,
        ConceptType.CONJUGATE_UPDATE,
        ConceptType.VARIATIONAL_INFERENCE,
        ConceptType.PRIOR_INIT,
    }
)

# Message-passing concept types that get memoized witness templates
_MESSAGE_PASSING_CONCEPT_TYPES = frozenset({ConceptType.MESSAGE_PASSING})

_SAFE_ANNOTATION_NAMES = frozenset(
    {
        "Any",
        "None",
        "bool",
        "bytes",
        "complex",
        "dict",
        "float",
        "frozenset",
        "int",
        "list",
        "Literal",
        "Mapping",
        "object",
        "set",
        "Sequence",
        "str",
        "tuple",
        "type",
        "Iterable",
        "Callable",
    }
)

_SAFE_ANNOTATION_ATTRS = frozenset(
    {
        ("np", "ndarray"),
    }
)

_BARE_GENERIC_REPLACEMENTS = {
    "Callable": "Callable[..., Any]",
    "Iterable": "Iterable[Any]",
    "Mapping": "Mapping[Any, Any]",
    "Sequence": "Sequence[Any]",
}


def _normalize_annotation_text(type_desc: str) -> str:
    """Normalize free-form type text into a Python-like expression."""
    annotation = (type_desc or "").strip()
    if not annotation:
        return "object"
    annotation = re.sub(r"\bor\b", "|", annotation)
    annotation = re.sub(r"\s*\|\s*", " | ", annotation)
    annotation = annotation.replace("array-like", "object")
    annotation = annotation.replace("CV splitter", "object")
    annotation = annotation.replace("cv splitter", "object")
    annotation = annotation.replace("iterable", "Iterable")
    annotation = annotation.replace("classifier", "object")
    annotation = annotation.replace("Classifier", "object")
    annotation = annotation.replace("regressor", "object")
    annotation = annotation.replace("Regressor", "object")
    return re.sub(r"\s+", " ", annotation).strip()


def _annotation_attr_chain(node: ast.AST) -> tuple[str, ...] | None:
    parts: list[str] = []
    current = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
        return tuple(reversed(parts))
    return None


def _sanitize_annotation_node(
    node: ast.AST, *, allowed_names: set[str]
) -> ast.expr:
    if isinstance(node, ast.Name):
        replacement = _BARE_GENERIC_REPLACEMENTS.get(node.id)
        if replacement is not None:
            parsed = ast.parse(replacement, mode="eval")
            return parsed.body
        if node.id in _SAFE_ANNOTATION_NAMES or node.id in allowed_names:
            return node
        return ast.Name(id="object", ctx=ast.Load())
    if isinstance(node, ast.Attribute):
        chain = _annotation_attr_chain(node)
        if chain in _SAFE_ANNOTATION_ATTRS:
            return node
        return ast.Name(id="object", ctx=ast.Load())
    if isinstance(node, ast.Subscript):
        if isinstance(node.value, ast.Name) and node.value.id in _BARE_GENERIC_REPLACEMENTS:
            value = node.value
        else:
            value = _sanitize_annotation_node(node.value, allowed_names=allowed_names)
        if isinstance(value, ast.Name) and value.id == "object":
            return value
        slice_node = _sanitize_annotation_node(node.slice, allowed_names=allowed_names)
        return ast.Subscript(value=value, slice=slice_node, ctx=ast.Load())
    if isinstance(node, ast.Tuple):
        return ast.Tuple(
            elts=[
                _sanitize_annotation_node(elt, allowed_names=allowed_names)
                for elt in node.elts
            ],
            ctx=ast.Load(),
        )
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.BitOr):
        return ast.BinOp(
            left=_sanitize_annotation_node(node.left, allowed_names=allowed_names),
            op=ast.BitOr(),
            right=_sanitize_annotation_node(node.right, allowed_names=allowed_names),
        )
    if isinstance(node, ast.Constant):
        return node
    return ast.Name(id="object", ctx=ast.Load())


def _python_annotation_expr(type_desc: str, *, allowed_names: set[str] | None = None) -> str:
    """Return a mypy-safe Python annotation for generated code."""
    normalized = _normalize_annotation_text(type_desc)
    try:
        parsed = ast.parse(normalized, mode="eval")
    except SyntaxError:
        return "object"
    allowed = set(allowed_names or ())
    sanitized = _sanitize_annotation_node(parsed.body, allowed_names=allowed)
    return ast.unparse(sanitized) or "object"


def _emit_source_class_loader(class_name: str, source_file: str) -> list[str]:
    """Load the original Python class from a source file at runtime."""
    if not source_file:
        return [f"{class_name}: Any = object"]
    path_literal = repr(str(Path(source_file).resolve()))
    return [
        "import importlib.util",
        "",
        f"_SCIONA_SOURCE_FILE = {path_literal}",
        '_SCIONA_SOURCE_SPEC = importlib.util.spec_from_file_location("_sciona_ingest_source", _SCIONA_SOURCE_FILE)',
        'if _SCIONA_SOURCE_SPEC is None or _SCIONA_SOURCE_SPEC.loader is None:',
        '    raise ImportError(f"Unable to load source module from {_SCIONA_SOURCE_FILE}")',
        "_SCIONA_SOURCE_MODULE = importlib.util.module_from_spec(_SCIONA_SOURCE_SPEC)",
        "_SCIONA_SOURCE_SPEC.loader.exec_module(_SCIONA_SOURCE_MODULE)",
        f'{class_name}: Any = getattr(_SCIONA_SOURCE_MODULE, "{class_name}")',
    ]


# ---------------------------------------------------------------------------
# Opaque DL boundary witness generation
# ---------------------------------------------------------------------------


def _opaque_witness_fallback(atom: MacroAtomSpec) -> str:
    """Generate a shape-preserving default witness for an opaque atom."""
    fn_name = _snake_case(atom.name)
    witness_name = f"witness_{fn_name}"

    params = []
    for inp in atom.inputs:
        params.append(f"{inp.name}: AbstractArray")
    param_str = ", ".join(params) if params else ""

    lines = [
        f"def {witness_name}({param_str}) -> AbstractArray:",
        f'    """Ghost witness for opaque boundary: {atom.name}."""',
    ]
    if atom.inputs:
        first = atom.inputs[0].name
        lines.append(f'    return AbstractArray(shape={first}.shape, dtype="float32")')
    else:
        lines.append('    return AbstractArray(shape=(), dtype="float32")')

    return "\n".join(lines)


async def generate_opaque_witnesses(
    macro_atoms: list[MacroAtomSpec],
    dfg: RawDataFlowGraph,
    llm: LLMClient,
    *,
    shared_context: SharedContextStore | None = None,
    shared_context_metrics: SharedContextMetrics | None = None,
    context_namespace: str = "",
    context_budget_chars: int = 900,
    parallelism: int = 1,
) -> tuple[str, dict[str, str]]:
    """Generate AbstractArray-based witnesses for opaque DL atoms.

    Attempts LLM shape inference; falls back to shape-preserving default.

    Returns (source_code, name_mapping).
    """
    lines = [
        '"""Auto-generated ghost witnesses for opaque DL boundaries."""',
        "",
        "from __future__ import annotations",
        "",
        "import torch",
        "import jax",
        "import jax.numpy as jnp",
        "import haiku as hk",
        "",
        "import networkx as nx  # type: ignore",
        "",
        "try:",
        "    from ageoa.ghost.abstract import AbstractArray",
        "except ImportError:",
        "    pass",
        "",
    ]

    async def _render_atom(atom: MacroAtomSpec) -> tuple[str, str, str]:
        fn_name = _snake_case(atom.name)
        witness_name = f"witness_{fn_name}"

        # Attempt LLM-drafted witness
        mf = dfg.methods[0] if dfg.methods else None
        witness_body: str | None = None

        if mf and llm is not None:
            param_specs = ", ".join(f'"{p}: AbstractArray"' for p in mf.params)
            try:
                user_prompt = DRAFT_OPAQUE_WITNESS_USER.format(
                    class_name=dfg.class_name,
                    base_classes=", ".join(dfg.opaque_base_classes),
                    method_name=mf.name,
                    params=", ".join(mf.params),
                    return_type=mf.return_type or "Any",
                    docstring=mf.docstring or "(none)",
                    fn_name=fn_name,
                    param_specs=param_specs,
                    return_type_spec="AbstractArray",
                )
                if shared_context is not None and context_namespace:
                    records = await shared_context.search(
                        f"{context_namespace}/opaque_witness",
                        f"{atom.name} {atom.description}",
                        limit=3,
                    )
                    block = format_context_block(
                        "Shared Context",
                        records,
                        max_chars=context_budget_chars,
                        metrics=shared_context_metrics,
                    )
                    if block:
                        user_prompt += f"\n\n{block}"
                response = await select_llm(llm, INGESTER_OPAQUE_WITNESS).complete(
                    DRAFT_OPAQUE_WITNESS_SYSTEM, user_prompt
                )
                raw = extract_json(response)
                witness_body = raw.get("witness_body")
            except Exception as exc:
                logger.warning("LLM witness drafting failed for %s: %s", atom.name, exc)

        if witness_body:
            params = []
            for inp in atom.inputs:
                params.append(f"{inp.name}: AbstractArray")
            param_str = ", ".join(params) if params else ""
            block_lines = [
                f"def {witness_name}({param_str}) -> AbstractArray:",
                f'    """Ghost witness for opaque boundary: {atom.name}."""',
            ]
            for body_line in witness_body.strip().splitlines():
                block_lines.append(f"    {body_line}")
            block_lines.append("")
            if shared_context is not None and context_namespace:
                try:
                    await shared_context.put(
                        f"{context_namespace}/opaque_witness",
                        (
                            f"Opaque atom: {atom.name}\n"
                            f"Witness: {witness_name}\n"
                            f"Hint: prefer shape-preserving returns and AbstractArray typing"
                        ),
                        metadata={"atom_name": atom.name, "witness_name": witness_name},
                    )
                except Exception:
                    pass
            return atom.name, witness_name, "\n".join(block_lines)

        fallback_block = _opaque_witness_fallback(atom) + "\n"
        return atom.name, witness_name, fallback_block

    opaque_atoms = [atom for atom in macro_atoms if atom.is_opaque]
    if not opaque_atoms:
        return "\n".join(lines), {}

    par = max(1, parallelism)
    if par <= 1 or len(opaque_atoms) <= 1:
        rendered = [await _render_atom(atom) for atom in opaque_atoms]
    else:
        semaphore = asyncio.Semaphore(par)

        async def _run(atom: MacroAtomSpec) -> tuple[str, str, str]:
            async with semaphore:
                return await _render_atom(atom)

        rendered = list(await asyncio.gather(*[_run(a) for a in opaque_atoms]))

    name_map: dict[str, str] = {}
    for atom_name, witness_name, block in rendered:
        name_map[atom_name] = witness_name
        lines.append(block.rstrip("\n"))
        lines.append("")

    return "\n".join(lines), name_map


# ---------------------------------------------------------------------------
# State model generation
# ---------------------------------------------------------------------------


def generate_state_models(specs: list[StateModelSpec]) -> str:
    """Generate Pydantic BaseModel classes from state model specs.

    When a spec has a ``stochastic`` field, injects RNG key and MCMC trace
    fields with appropriate types and defaults.
    """
    if not specs:
        return ""

    allowed_names = {spec.model_name for spec in specs}
    has_stochastic = any(spec.stochastic is not None for spec in specs)

    lines = [
        '"""Auto-generated Pydantic state models for cross-window state."""',
        "",
        "from __future__ import annotations",
        "",
        "from typing import Any, Callable, Iterable, Literal, Mapping, Sequence",
        "",
        "from pydantic import BaseModel, ConfigDict, Field",
        "",
    ]
    if has_stochastic:
        lines.extend(
            [
                "import numpy as np",
                "",
            ]
        )

    for spec in specs:
        if spec.docstring:
            lines.append(f"class {spec.model_name}(BaseModel):")
            lines.append(f'    """{spec.docstring}"""')
        else:
            lines.append(f"class {spec.model_name}(BaseModel):")

        lines.append("    model_config = ConfigDict(arbitrary_types_allowed=True)")
        lines.append("")
        if not spec.fields and spec.stochastic is None:
            lines.append("    pass")
        else:
            for field_name, field_type in spec.fields:
                annotation = _python_annotation_expr(
                    field_type, allowed_names=allowed_names
                )
                lines.append(
                    f"    {field_name}: {annotation} | None = Field(default=None)"
                )

            # Inject stochastic state fields
            if spec.stochastic is not None:
                st = spec.stochastic
                lines.append("")
                lines.append("    # --- Stochastic state (auto-generated) ---")
                lines.append(f"    {st.rng_field}: Any = Field(")
                lines.append("        default=None,")
                lines.append(
                    f'        description="RNG state ({st.rng_type}). '
                    f'Split before each stochastic atom.",'
                )
                lines.append("    )")

                if st.trace_field:
                    dims_str = str(st.trace_param_dims)
                    lines.append(f"    {st.trace_field}: Any = Field(")
                    lines.append("        default=None,")
                    lines.append(
                        f'        description="MCMC trace. '
                        f"param_dims={dims_str}, "
                        f"chains={st.chain_count}, "
                        f'warmup={st.warmup_steps}",'
                    )
                    lines.append("    )")
                    lines.append("    mcmc_step_count: int = Field(default=0)")
                    lines.append("    mcmc_accept_rate: float = Field(default=0.0)")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Atom wrapper generation
# ---------------------------------------------------------------------------


def _snake_case(name: str) -> str:
    """Convert a name like 'Signal Conditioner' to 'signal_conditioner'."""
    return name.lower().replace(" ", "_").replace("-", "_")


def _canonical_ir(plan: ValidatedMacroPlan | None) -> IngestIRPlan | None:
    if plan is None:
        return None
    return plan.plan.canonical_ir


def _canonical_group_for_atom(
    plan: ValidatedMacroPlan | None,
    atom: MacroAtomSpec,
) -> PlannedOperationGroup | None:
    if plan is None or plan.plan.planning_graph is None:
        return None
    atom_id = _snake_case(atom.name)
    for group in plan.plan.planning_graph.planned_groups:
        if group.group_id == atom_id or _snake_case(group.display_name) == atom_id:
            return group
    return None


def _canonical_operation_for_atom(
    plan: ValidatedMacroPlan | None,
    atom: MacroAtomSpec,
) -> tuple[OperationSpec | None, PlannedOperationGroup | None]:
    ir = _canonical_ir(plan)
    if ir is None:
        return None, None

    group = _canonical_group_for_atom(plan, atom)
    operations = {operation.operation_id: operation for operation in ir.operations}
    if group is not None:
        for operation_id in group.member_operation_ids:
            operation = operations.get(operation_id)
            if operation is not None:
                return operation, group

    atom_id = _snake_case(atom.name)
    for operation in ir.operations:
        if operation.operation_id == atom_id or _snake_case(operation.display_name) == atom_id:
            return operation, group

    atom_methods = set(atom.method_names)
    if atom_methods:
        for operation in ir.operations:
            binding_names = {binding.method_name for binding in operation.method_bindings}
            if atom_methods.issubset(binding_names):
                return operation, group

    return None, group


def _canonical_state_slot_map(ir: IngestIRPlan | None) -> dict[str, StateSlotSpec]:
    if ir is None:
        return {}
    return {slot.slot_name: slot for slot in ir.state_slots}


def _state_model_field_map(state_models: list[StateModelSpec]) -> dict[str, str]:
    fields: dict[str, str] = {}
    for state_model in state_models:
        for field_name, field_type in state_model.fields:
            fields.setdefault(field_name, field_type)
    return fields


def _canonical_method_bindings_for_atom(
    atom: MacroAtomSpec,
    operation: OperationSpec,
    group: PlannedOperationGroup | None,
) -> list[MethodBinding]:
    if not operation.method_bindings:
        return []

    selected_names = set(atom.method_names)
    if not selected_names and group is not None and "__" in group.group_id:
        suffix = group.group_id.split("__", 1)[1]
        selected_names = {
            binding.method_name
            for binding in operation.method_bindings
            if _snake_case(binding.method_name) == suffix
        }
    if not selected_names and group is not None:
        display_key = _snake_case(group.display_name)
        selected_names = {
            binding.method_name
            for binding in operation.method_bindings
            if _snake_case(binding.method_name) == display_key
        }

    if not selected_names:
        return list(operation.method_bindings)

    matched = [
        binding
        for binding in operation.method_bindings
        if binding.method_name in selected_names
        or _snake_case(binding.method_name) in selected_names
    ]
    return matched or list(operation.method_bindings)


def _canonical_effects_for_methods(
    operation: OperationSpec,
    method_names: set[str],
) -> list[StateEffectSpec]:
    if not method_names:
        return list(operation.state_effects)
    filtered = [
        effect
        for effect in operation.state_effects
        if not effect.source_method or effect.source_method in method_names
    ]
    return filtered


def _canonical_output_bindings_for_atom(
    atom: MacroAtomSpec,
    operation: OperationSpec,
    method_names: set[str],
    group: PlannedOperationGroup | None,
) -> list[OutputBindingSpec]:
    bindings = list(group.emitted_outputs) if group is not None and group.emitted_outputs else list(operation.emitted_outputs)
    if method_names:
        bindings = [
            binding
            for binding in bindings
            if not binding.source_method or binding.source_method in method_names
        ]
    if not atom.outputs:
        return []
    allowed_output_names = {output.name for output in atom.outputs}
    return [binding for binding in bindings if binding.output_name in allowed_output_names]


def _canonical_required_state_slots(
    operation: OperationSpec,
    method_names: set[str],
    slot_map: dict[str, StateSlotSpec],
    effects: list[StateEffectSpec],
) -> list[str]:
    effect_slots = {effect.slot_name for effect in effects}
    slots: list[str] = []
    for slot_name in operation.required_state_slots:
        slot = slot_map.get(slot_name)
        if not method_names or slot is None:
            slots.append(slot_name)
            continue
        reads = set(slot.read_by)
        writes = set(slot.written_by)
        if reads.intersection(method_names) or writes.intersection(method_names) or slot_name in effect_slots:
            slots.append(slot_name)
    for slot_name in sorted(effect_slots):
        if slot_name not in slots:
            slots.append(slot_name)
    return slots


def _canonical_wrapper_inputs(
    atom: MacroAtomSpec,
    operation: OperationSpec,
    bindings: list[MethodBinding],
    slot_map: dict[str, StateSlotSpec],
    state_model_fields: dict[str, str],
    required_slots: list[str],
) -> list[IOSpec]:
    binding_param_names = {
        param.name
        for binding in bindings
        for param in binding.signature
        if param.name != "self"
    }
    base_inputs = list(operation.direct_inputs or atom.inputs)
    filtered_inputs: list[IOSpec] = []
    seen: set[str] = set()
    for spec in base_inputs:
        if binding_param_names and spec.name not in binding_param_names:
            continue
        filtered_inputs.append(spec)
        seen.add(spec.name)

    for slot_name in required_slots:
        slot = slot_map.get(slot_name)
        if slot is None or slot.state_kind != "config":
            continue
        if slot_name in seen or slot_name in state_model_fields:
            continue
        filtered_inputs.append(IOSpec(name=slot_name, type_desc=slot.type_desc or "Any"))
        seen.add(slot_name)
    return filtered_inputs


def _canonical_call_arguments(
    binding: MethodBinding,
    available_inputs: set[str],
) -> tuple[list[str], str | None]:
    if not binding.signature:
        return [], None

    args: list[str] = []
    for param in binding.signature:
        if param.name == "self":
            continue
        if param.name not in available_inputs:
            if param.has_default:
                continue
            return [], f"missing required parameter {param.name!r} for {binding.method_name}"
        if param.kind in {"positional_only", "positional_or_keyword", ""}:
            args.append(param.name)
        elif param.kind == "keyword_only":
            args.append(f"{param.name}={param.name}")
        elif param.kind == "vararg":
            args.append(f"*{param.name}")
        elif param.kind == "kwarg":
            args.append(f"**{param.name}")
        else:
            return [], f"unsupported parameter kind {param.kind!r} for {binding.method_name}"
    return args, None


def _canonical_output_expression(
    binding: OutputBindingSpec,
    return_vars: dict[str, str],
    fallback_return_var: str,
    *,
    source_language: str = "python",
) -> tuple[str | None, str | None]:
    if binding.binding_kind == "self_return":
        return None, None

    source_var = return_vars.get(binding.source_method, fallback_return_var)
    if binding.binding_kind in {"return_value", "metadata_object"}:
        return source_var, None
    if binding.binding_kind == "attribute_read":
        if source_language != "python":
            return None, (
                f"attribute_read binding for {binding.output_name} is unsupported for "
                f"non-python source {source_language!r}"
            )
        if not binding.source_attr:
            return None, f"attribute_read binding for {binding.output_name} has no source_attr"
        return f"obj.{binding.source_attr}", None
    if binding.binding_kind == "tuple_element":
        if binding.tuple_index is None:
            return None, f"tuple_element binding for {binding.output_name} has no tuple_index"
        return f"{source_var}[{binding.tuple_index}]", None
    if binding.binding_kind == "constant":
        if not binding.source_attr:
            return None, f"constant binding for {binding.output_name} has no literal expression"
        return binding.source_attr, None
    return None, f"unsupported binding kind {binding.binding_kind!r} for {binding.output_name}"


def _canonical_return_annotation(
    atom: MacroAtomSpec,
    output_bindings: list[OutputBindingSpec],
    *,
    allowed_names: set[str],
) -> str:
    if not output_bindings:
        return "None"
    annotations = []
    outputs_by_name = {output.name: output for output in atom.outputs}
    for binding in output_bindings:
        output_spec = outputs_by_name.get(binding.output_name)
        type_desc = binding.type_desc or (output_spec.type_desc if output_spec is not None else "Any")
        annotations.append(_python_annotation_expr(type_desc, allowed_names=allowed_names))
    if len(annotations) == 1:
        return annotations[0]
    return "tuple[" + ", ".join(annotations) + "]"


def _emit_docstring(
    lines: list[str],
    atom: MacroAtomSpec,
    *,
    state_type: str | None = None,
    return_annotation: str,
) -> None:
    summary = atom.description or f"Compute {atom.name}."
    lines.append(f'    """{summary}')
    lines.append("")
    if atom.inputs or state_type is not None:
        lines.append("    Args:")
        for inp in atom.inputs:
            lines.append(f"        {inp.name}: {inp.constraints or 'Input data.'}")
        if state_type is not None:
            lines.append(
                f"        state: {state_type} object containing cross-window persistent state."
            )
    lines.append("")
    lines.append("    Returns:")
    lines.append(f"        {return_annotation}")
    lines.append('    """')


def _emit_canonical_wrapper_body(
    lines: list[str],
    *,
    atom: MacroAtomSpec,
    bindings: list[MethodBinding],
    output_bindings: list[OutputBindingSpec],
    wrapper_inputs: list[IOSpec],
    required_slots: list[str],
    effects: list[StateEffectSpec],
    state_model_fields: dict[str, str],
    slot_map: dict[str, StateSlotSpec],
    class_name: str,
    stateful: bool,
    state_type: str | None,
    source_language: str,
) -> None:
    if not bindings:
        raise ValueError(f"{atom.name}: canonical operation has no method bindings")

    wrapper_input_names = {spec.name for spec in wrapper_inputs}
    if source_language == "python":
        lines.append(f"    obj = {class_name}.__new__({class_name})")
        for slot_name in required_slots:
            slot = slot_map.get(slot_name)
            if slot_name in state_model_fields and stateful:
                lines.append(f"    obj.{slot_name} = state.{slot_name}")
                continue
            if slot is not None and slot.state_kind == "config" and slot_name in wrapper_input_names:
                lines.append(f"    obj.{slot_name} = {slot_name}")
                continue
            raise ValueError(f"{atom.name}: no canonical source for required state slot {slot_name!r}")
    elif required_slots or stateful:
        raise ValueError(
            f"{atom.name}: canonical non-python emission does not support required state"
        )

    return_vars: dict[str, str] = {}
    available_inputs = set(wrapper_input_names)
    expected_inputs = [spec.name for spec in wrapper_inputs]
    for index, binding in enumerate(bindings):
        call_args, error = _canonical_call_arguments(binding, available_inputs)
        if error is not None:
            raise ValueError(f"{atom.name}: {error}")
        return_var = f"_ret_{index}"
        return_vars[binding.method_name] = return_var
        if source_language == "python":
            call_expr = ", ".join(call_args)
            lines.append(f"    {return_var} = obj.{binding.method_name}({call_expr})")
        else:
            if len(bindings) > 1:
                raise ValueError(
                    f"{atom.name}: canonical non-python emission requires a single binding"
                )
            if call_args != expected_inputs:
                raise ValueError(
                    f"{atom.name}: canonical non-python emission requires direct ffi-call inputs"
                )
            lines.append(
                f"    {return_var} = {_snake_case(atom.name)}_ffi({', '.join(expected_inputs)})"
            )

    update_slots: list[str] = []
    for effect in effects:
        if effect.effect_kind not in {"initialize", "update", "clear"}:
            continue
        if effect.slot_name in state_model_fields and effect.slot_name not in update_slots:
            update_slots.append(effect.slot_name)

    if stateful:
        if update_slots:
            lines.append("    new_state = state.model_copy(update={")
            for slot_name in update_slots:
                lines.append(f'        "{slot_name}": obj.{slot_name},')
            lines.append("    })")
        else:
            lines.append("    new_state = state")

    value_expressions: list[str] = []
    fallback_return_var = return_vars[bindings[-1].method_name]
    for binding in output_bindings:
        expression, error = _canonical_output_expression(
            binding,
            return_vars,
            fallback_return_var,
            source_language=source_language,
        )
        if error is not None:
            raise ValueError(f"{atom.name}: {error}")
        if expression is not None:
            value_expressions.append(expression)

    if atom.outputs and len(value_expressions) != len(atom.outputs):
        raise ValueError(
            f"{atom.name}: canonical bindings resolved {len(value_expressions)} outputs for {len(atom.outputs)} declared outputs"
        )

    if not value_expressions:
        result_expr = "None"
    elif len(value_expressions) == 1:
        result_expr = value_expressions[0]
    else:
        result_expr = "(" + ", ".join(value_expressions) + ")"

    if stateful:
        lines.append(f"    return {result_expr}, new_state")
    else:
        lines.append(f"    return {result_expr}")


def _canonical_emission_context(
    atom: MacroAtomSpec,
    *,
    state_models: list[StateModelSpec],
    plan: ValidatedMacroPlan | None,
    source_language: str,
) -> dict[str, object] | None:
    if atom.concept_type in _MESSAGE_PASSING_CONCEPT_TYPES:
        return None

    canonical_ir = _canonical_ir(plan)
    if canonical_ir is None:
        return None

    operation, group = _canonical_operation_for_atom(plan, atom)
    if operation is None:
        return None

    slot_map = _canonical_state_slot_map(canonical_ir)
    state_model_fields = _state_model_field_map(state_models)
    operations = [operation]
    if group is not None and len(group.member_operation_ids) > 1:
        operations_by_id = {item.operation_id: item for item in canonical_ir.operations}
        ordered_operations = [
            operations_by_id[operation_id]
            for operation_id in group.member_operation_ids
            if operation_id in operations_by_id
        ]
        if ordered_operations:
            operations = ordered_operations

    bindings: list[MethodBinding] = []
    effects: list[StateEffectSpec] = []
    effect_keys: set[tuple[str, str, str]] = set()
    required_slots: list[str] = []
    required_slot_set: set[str] = set()
    direct_inputs: list[IOSpec] = []
    direct_input_names: set[str] = set()
    aggregated_outputs: list[OutputBindingSpec] = []
    output_keys: set[tuple[str, str, str, str, int | None]] = set()
    all_method_names: set[str] = set()

    for grouped_operation in operations:
        op_group = group if len(operations) == 1 else None
        op_bindings = _canonical_method_bindings_for_atom(atom, grouped_operation, op_group)
        bindings.extend(op_bindings)
        method_names = {binding.method_name for binding in op_bindings}
        all_method_names.update(method_names)

        for slot_name in grouped_operation.required_state_slots:
            if slot_name in required_slot_set:
                continue
            required_slots.append(slot_name)
            required_slot_set.add(slot_name)

        for effect in _canonical_effects_for_methods(grouped_operation, method_names):
            key = (effect.slot_name, effect.effect_kind, effect.source_method)
            if key in effect_keys:
                continue
            effects.append(effect)
            effect_keys.add(key)

        slot_candidates = _canonical_required_state_slots(
            grouped_operation,
            method_names,
            slot_map,
            effects,
        )
        for slot_name in slot_candidates:
            if slot_name in required_slot_set:
                continue
            required_slots.append(slot_name)
            required_slot_set.add(slot_name)

        for spec in grouped_operation.direct_inputs:
            if spec.name in direct_input_names:
                continue
            direct_inputs.append(spec)
            direct_input_names.add(spec.name)

        for output in _canonical_output_bindings_for_atom(atom, grouped_operation, method_names, None):
            key = (
                output.output_name,
                output.binding_kind,
                output.source_method,
                output.source_attr,
                output.tuple_index,
            )
            if key in output_keys:
                continue
            aggregated_outputs.append(output)
            output_keys.add(key)

    if group is not None and group.required_state_slots:
        for slot_name in group.required_state_slots:
            if slot_name in required_slot_set:
                continue
            required_slots.append(slot_name)
            required_slot_set.add(slot_name)

    if group is not None and group.emitted_outputs:
        allowed_output_names = {output.name for output in atom.outputs}
        group_outputs: list[OutputBindingSpec] = []
        seen_group_outputs: set[tuple[str, str, str, str, int | None]] = set()
        for output in group.emitted_outputs:
            if allowed_output_names and output.output_name not in allowed_output_names:
                continue
            if output.source_method and all_method_names and output.source_method not in all_method_names:
                continue
            key = (
                output.output_name,
                output.binding_kind,
                output.source_method,
                output.source_attr,
                output.tuple_index,
            )
            if key in seen_group_outputs:
                continue
            group_outputs.append(output)
            seen_group_outputs.add(key)
        if group_outputs:
            aggregated_outputs = group_outputs

    binding_param_names = {
        param.name
        for binding in bindings
        for param in binding.signature
        if param.name != "self"
    }
    base_inputs = direct_inputs or list(operation.direct_inputs or atom.inputs)
    if not base_inputs:
        base_inputs = list(atom.inputs)

    wrapper_inputs: list[IOSpec] = []
    wrapper_input_names: set[str] = set()
    for spec in base_inputs:
        if binding_param_names and spec.name not in binding_param_names:
            continue
        if spec.name in wrapper_input_names:
            continue
        wrapper_inputs.append(spec)
        wrapper_input_names.add(spec.name)

    for slot_name in required_slots:
        slot = slot_map.get(slot_name)
        if slot is None or slot.state_kind != "config":
            continue
        if slot_name in wrapper_input_names or slot_name in state_model_fields:
            continue
        wrapper_inputs.append(IOSpec(name=slot_name, type_desc=slot.type_desc or "Any"))
        wrapper_input_names.add(slot_name)

    return {
        "canonical_ir": canonical_ir,
        "operation": operation,
        "group": group,
        "bindings": bindings,
        "effects": effects,
        "output_bindings": aggregated_outputs,
        "required_slots": required_slots,
        "wrapper_inputs": wrapper_inputs,
        "slot_map": slot_map,
        "state_model_fields": state_model_fields,
    }


def generate_atom_wrappers(
    macro_atoms: list[MacroAtomSpec],
    state_models: list[StateModelSpec],
    witness_names: dict[str, str],
    source_language: str = "python",
    class_name: str = "",
    source_file: str = "",
    plan: ValidatedMacroPlan | None = None,
) -> str:
    """Generate ``@register_atom`` decorated function wrappers."""
    allowed_names = {spec.model_name for spec in state_models}
    canonical_ir = _canonical_ir(plan)
    lines = [
        '"""Auto-generated atom wrappers following the ageoa pattern."""',
        "",
        "from __future__ import annotations",
        "",
        "# mypy: disable-error-code=untyped-decorator",
        "",
        "from typing import Any, Callable, Iterable, Literal, Mapping, Sequence",
        "",
        "import numpy as np",
        "import icontract",
        "from ageoa.ghost.registry import register_atom",
        "",
    ]

    if canonical_ir is not None and source_language == "python" and class_name:
        lines.extend(_emit_source_class_loader(class_name, source_file))
        lines.append("")

    # Add FFI imports for non-Python sources
    if source_language != "python":
        lines.append(generate_ffi_imports(source_language))
        lines.append("")

    # Import state models if any
    if state_models:
        names = ", ".join(spec.model_name for spec in state_models)
        lines.append(f"from state_models import {names}")
        lines.append("")

    # Import witness functions
    if witness_names:
        names = ", ".join(sorted(set(witness_names.values())))
        lines.append(f"from witnesses import {names}")
        lines.append("")

    # Memoization preamble for message-passing atoms
    has_message_passing = any(
        a.concept_type in _MESSAGE_PASSING_CONCEPT_TYPES for a in macro_atoms
    )
    if has_message_passing:
        lines.extend(
            [
                "_MEMO: dict = {}",
                "",
                "",
                "def _memo_key(name: str, *args) -> tuple:",
                '    """Build a memoization cache key from name and argument ids."""',
                "    return (name,) + tuple(id(a) for a in args)",
                "",
                "",
            ]
        )

    for atom in macro_atoms:
        fn_name = _snake_case(atom.name)
        witness_fn = witness_names.get(atom.name, f"witness_{fn_name}")

        canonical_context = _canonical_emission_context(
            atom,
            state_models=state_models,
            plan=plan,
            source_language=source_language,
        )
        canonical_operation = None
        canonical_bindings: list[MethodBinding] = []
        canonical_outputs: list[OutputBindingSpec] = []
        canonical_wrapper_inputs: list[IOSpec] = []
        if canonical_context is not None:
            canonical_operation = canonical_context["operation"]
            canonical_bindings = canonical_context["bindings"]
            canonical_outputs = canonical_context["output_bindings"]
            canonical_wrapper_inputs = canonical_context["wrapper_inputs"]

        # Build parameter list
        params = []
        wrapper_inputs = canonical_wrapper_inputs or list(atom.inputs)
        for inp in wrapper_inputs:
            annotation = _python_annotation_expr(
                inp.type_desc, allowed_names=allowed_names
            )
            params.append(f"{inp.name}: {annotation}")
        param_str = ", ".join(params) if params else ""

        # Build return type
        if canonical_operation is not None:
            ret_type = _canonical_return_annotation(
                atom,
                canonical_outputs,
                allowed_names=allowed_names,
            )
        elif atom.outputs:
            if len(atom.outputs) == 1:
                ret_type = _python_annotation_expr(
                    atom.outputs[0].type_desc, allowed_names=allowed_names
                )
            else:
                output_annotations = [
                    _python_annotation_expr(o.type_desc, allowed_names=allowed_names)
                    for o in atom.outputs
                ]
                ret_type = "tuple[" + ", ".join(output_annotations) + "]"
        else:
            ret_type = "None"

        # Outermost: @register_atom (runs last)
        lines.append(f"@register_atom({witness_fn})")

        # 1. Add LLM-provided decorators
        for deco in getattr(atom, "decorators", []):
            lines.append(deco)

        # 2. Add default contracts (if missing) to satisfy the rule:
        # "At least one @require and one @ensure per atom."
        has_require = any("@icontract.require" in d for d in getattr(atom, "decorators", []))
        has_ensure = any("@icontract.ensure" in d for d in getattr(atom, "decorators", []))

        if not has_require:
            # Add a basic type check for each input
            for inp in wrapper_inputs:
                if "np.ndarray" in inp.type_desc or "AbstractArray" in inp.type_desc:
                    lines.append(f'@icontract.require(lambda {inp.name}: isinstance({inp.name}, np.ndarray), "{inp.name} must be a numpy array")')
                elif "float" in inp.type_desc:
                    lines.append(f'@icontract.require(lambda {inp.name}: isinstance({inp.name}, (float, int, np.number)), "{inp.name} must be numeric")')
            # If still no require, add a generic one
            if not any("@icontract.require" in d for d in lines[-len(wrapper_inputs)-1:]):
                for inp in wrapper_inputs:
                    lines.append(f'@icontract.require(lambda {inp.name}: {inp.name} is not None, "{inp.name} cannot be None")')

        if not has_ensure:
            if atom.outputs:
                if len(atom.outputs) == 1:
                    lines.append(f'@icontract.ensure(lambda result, **kwargs: result is not None, "{atom.name} output must not be None")')
                else:
                    lines.append(f'@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "{atom.name} all outputs must not be None")')

        lines.append(f"def {fn_name}({param_str}) -> {ret_type}:")

        # Message-passing atoms get memoized wrappers
        if atom.concept_type in _MESSAGE_PASSING_CONCEPT_TYPES:
            desc = atom.description or f"Compute {atom.name} with memoization."
            # Google-style docstring
            lines.append(f'    """{desc}')
            lines.append("")
            if atom.inputs:
                lines.append("    Args:")
                for inp in atom.inputs:
                    lines.append(f"        {inp.name}: {inp.constraints or 'Input data.'}")
            if atom.outputs:
                lines.append("")
                lines.append("    Returns:")
                if len(atom.outputs) == 1:
                    lines.append(f"        {atom.outputs[0].constraints or 'Result data.'}")
                else:
                    for out in atom.outputs:
                        lines.append(f"        {out.name}: {out.constraints or 'Result data.'}")
            lines.append('    """')
            arg_names = ", ".join(inp.name for inp in atom.inputs)
            lines.append(f'    _key = _memo_key("{fn_name}", {arg_names})')
            lines.append("    if _key in _MEMO:")
            lines.append("        return _MEMO[_key]")
            lines.append(
                '    raise NotImplementedError("Wire to original implementation")'
            )
        else:
            if canonical_operation is not None:
                _emit_docstring(
                    lines,
                    atom,
                    return_annotation=ret_type,
                )
                try:
                    _emit_canonical_wrapper_body(
                        lines,
                        atom=atom,
                        bindings=canonical_bindings,
                        output_bindings=canonical_outputs,
                        wrapper_inputs=wrapper_inputs,
                        required_slots=canonical_context["required_slots"],
                        effects=canonical_context["effects"],
                        state_model_fields=canonical_context["state_model_fields"],
                        slot_map=canonical_context["slot_map"],
                        class_name=class_name or "object",
                        stateful=False,
                        state_type=None,
                        source_language=source_language,
                    )
                except ValueError as exc:
                    lines.append(f'    raise NotImplementedError("{str(exc).replace(chr(34), chr(39))}")')
            else:
                if atom.description:
                    # Google-style docstring
                    lines.append(f'    """{atom.description}')
                    lines.append("")
                    if atom.inputs:
                        lines.append("    Args:")
                        for inp in atom.inputs:
                            lines.append(f"        {inp.name}: {inp.constraints or 'Input data.'}")
                    if atom.outputs:
                        lines.append("")
                        lines.append("    Returns:")
                        if len(atom.outputs) == 1:
                            lines.append(f"        {atom.outputs[0].constraints or 'Result data.'}")
                        else:
                            for out in atom.outputs:
                                lines.append(f"        {out.name}: {out.constraints or 'Result data.'}")
                    lines.append('    """')
                lines.append(
                    '    raise NotImplementedError("Wire to original implementation")'
                )
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Stateful wrapper generation
# ---------------------------------------------------------------------------


def generate_stateful_wrappers(
    macro_atoms: list[MacroAtomSpec],
    state_models: list[StateModelSpec],
    class_name: str,
    witness_names: dict[str, str],
    source_file: str = "",
    source_language: str = "python",
    plan: ValidatedMacroPlan | None = None,
) -> str:
    """Generate ``@register_atom`` wrappers with inject/run/extract state pattern.

    Each wrapper instantiates the legacy class via ``__new__``, injects ALL
    state fields, runs the original method(s), and extracts ALL state fields
    back into an immutable ``model_copy`` update.
    """
    if not state_models:
        return generate_atom_wrappers(
            macro_atoms,
            state_models,
            witness_names,
            source_language=source_language,
            class_name=class_name,
            source_file=source_file,
            plan=plan,
        )

    state_model = state_models[0]
    state_type = state_model.model_name
    allowed_names = {spec.model_name for spec in state_models}

    lines = [
        '"""Auto-generated stateful atom wrappers following the ageoa pattern."""',
        "",
        "from __future__ import annotations",
        "",
        "# mypy: disable-error-code=untyped-decorator",
        "",
        "from typing import Any, Callable, Iterable, Literal, Mapping, Sequence",
        "",
        "import numpy as np",
        "import icontract",
        "from ageoa.ghost.registry import register_atom",
        "",
    ]

    if source_language == "python":
        lines.extend(_emit_source_class_loader(class_name, source_file))
        lines.append("")
    else:
        lines.append(f"{class_name}: Any = object")
        lines.append("")

    state_model_names = ", ".join(spec.model_name for spec in state_models)
    lines.append(f"from state_models import {state_model_names}")
    lines.append("")

    # Import witness functions
    if witness_names:
        names = ", ".join(sorted(set(witness_names.values())))
        lines.append(f"from witnesses import {names}")
        lines.append("")

    for atom in macro_atoms:
        fn_name = _snake_case(atom.name)
        witness_fn = witness_names.get(atom.name, f"witness_{fn_name}")
        canonical_context = _canonical_emission_context(
            atom,
            state_models=state_models,
            plan=plan,
            source_language=source_language,
        )

        # Build parameter list — original params + state
        params = []
        wrapper_inputs = (
            canonical_context["wrapper_inputs"]
            if canonical_context is not None
            else list(atom.inputs)
        )
        for inp in wrapper_inputs:
            annotation = _python_annotation_expr(
                inp.type_desc, allowed_names=allowed_names
            )
            params.append(f"{inp.name}: {annotation}")
        params.append(f"state: {state_type}")
        param_str = ", ".join(params)

        # Build return type — (original_return, StateType)
        if canonical_context is not None:
            orig_ret = _canonical_return_annotation(
                atom,
                canonical_context["output_bindings"],
                allowed_names=allowed_names,
            )
        elif atom.outputs:
            if len(atom.outputs) == 1:
                orig_ret = _python_annotation_expr(
                    atom.outputs[0].type_desc, allowed_names=allowed_names
                )
            else:
                output_annotations = [
                    _python_annotation_expr(o.type_desc, allowed_names=allowed_names)
                    for o in atom.outputs
                ]
                orig_ret = "tuple[" + ", ".join(output_annotations) + "]"
        else:
            orig_ret = "None"
        ret_type = f"tuple[{orig_ret}, {state_type}]"

        # Outermost: @register_atom (runs last)
        lines.append(f"@register_atom({witness_fn})")

        # LLM-provided + Default contracts
        has_require = any("@icontract.require" in d for d in getattr(atom, "decorators", []))
        has_ensure = any("@icontract.ensure" in d for d in getattr(atom, "decorators", []))

        for deco in getattr(atom, "decorators", []):
            lines.append(deco)

        if not has_require:
            for inp in wrapper_inputs:
                if "np.ndarray" in inp.type_desc or "AbstractArray" in inp.type_desc:
                    lines.append(f'@icontract.require(lambda {inp.name}: isinstance({inp.name}, np.ndarray), "{inp.name} must be a numpy array")')
                elif "float" in inp.type_desc:
                    lines.append(f'@icontract.require(lambda {inp.name}: isinstance({inp.name}, (float, int, np.number)), "{inp.name} must be numeric")')
            if not any("@icontract.require" in d for d in lines[-len(wrapper_inputs)-1:]):
                for inp in wrapper_inputs:
                    lines.append(f'@icontract.require(lambda {inp.name}: {inp.name} is not None, "{inp.name} cannot be None")')

        if not has_ensure:
            if atom.outputs:
                if len(atom.outputs) == 1:
                    lines.append(f'@icontract.ensure(lambda result, **kwargs: result is not None, "{atom.name} output must not be None")')
                else:
                    lines.append(f'@icontract.ensure(lambda result, **kwargs: all(r is not None for r in result), "{atom.name} all outputs must not be None")')

        lines.append(f"def {fn_name}({param_str}) -> {ret_type}:")

        if canonical_context is not None:
            _emit_docstring(
                lines,
                atom,
                state_type=state_type,
                return_annotation=ret_type,
            )
            try:
                _emit_canonical_wrapper_body(
                    lines,
                    atom=atom,
                    bindings=canonical_context["bindings"],
                    output_bindings=canonical_context["output_bindings"],
                    wrapper_inputs=wrapper_inputs,
                    required_slots=canonical_context["required_slots"],
                    effects=canonical_context["effects"],
                    state_model_fields=canonical_context["state_model_fields"],
                    slot_map=canonical_context["slot_map"],
                    class_name=class_name or "object",
                    stateful=True,
                    state_type=state_type,
                    source_language=source_language,
                )
            except ValueError as exc:
                lines.append(f'    raise NotImplementedError("{str(exc).replace(chr(34), chr(39))}")')
        else:
            # Google-style docstring
            summary = atom.description or f"Compute {atom.name}."
            lines.append('    """Stateless wrapper: Functional Core, Imperative Shell.')
            lines.append("")
            lines.append(f"    {summary}")
            lines.append("")
            if atom.inputs:
                lines.append("    Args:")
                for inp in atom.inputs:
                    lines.append(f"        {inp.name}: {inp.constraints or 'Input data.'}")
            lines.append(f"        state: {state_type} object containing cross-window persistent state.")
            if atom.outputs:
                lines.append("")
                lines.append("    Returns:")
                if len(atom.outputs) == 1:
                    lines.append(f"        tuple[{atom.outputs[0].constraints or 'Result'}, {state_type}]:")
                else:
                    lines.append(f"        tuple[tuple[{', '.join(o.name for o in atom.outputs)}], {state_type}]:")
                lines.append(f"            The first element is the functional result, the second is the updated state.")
            lines.append('    """')

            # Instantiate via __new__
            lines.append(f"    obj = {class_name}.__new__({class_name})")

            # Inject ALL state fields
            for field_name, _ in state_model.fields:
                lines.append(f"    obj.{field_name} = state.{field_name}")

            # Run method(s)
            for method_name in atom.method_names:
                if method_name == "__init__":
                    continue
                # Build call args from atom inputs
                call_args = ", ".join(inp.name for inp in atom.inputs)
                lines.append(f"    obj.{method_name}({call_args})")

            # Extract ALL state fields via model_copy
            lines.append("    new_state = state.model_copy(update={")
            for fname, _ in state_model.fields:
                lines.append(f'        "{fname}": obj.{fname},')
            lines.append("    })")

            # Build return value
            if atom.outputs:
                if len(atom.outputs) == 1:
                    out = atom.outputs[0]
                    lines.append(f"    result = obj.{out.name}")
                    lines.append("    return result, new_state")
                else:
                    out_names = ", ".join(f"obj.{o.name}" for o in atom.outputs)
                    lines.append(f"    result = ({out_names})")
                    lines.append("    return result, new_state")
            else:
                lines.append("    return None, new_state")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Ghost witness generation
# ---------------------------------------------------------------------------


def _generate_bayesian_witness(
    atom: MacroAtomSpec,
    fn_name: str,
    witness_name: str,
    has_state: bool,
) -> list[str]:
    """Generate a specialized witness for a Bayesian atom.

    Routes to the appropriate witness template based on concept_type.
    """
    lines: list[str] = []
    ct = atom.concept_type

    if ct == ConceptType.PRIOR_INIT:
        params = ["event_shape: tuple[int, ...]", 'family: str = "normal"']
        lines.append(
            f"def {witness_name}({', '.join(params)}) -> AbstractDistribution:"
        )
        lines.append(f'    """Ghost witness for prior init: {atom.name}."""')
        lines.append("    return AbstractDistribution(")
        lines.append("        family=family,")
        lines.append("        event_shape=event_shape,")
        lines.append("    )")

    elif ct == ConceptType.LOG_PROB:
        params = ["dist: AbstractDistribution", "samples: AbstractArray"]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractScalar:")
        lines.append(f'    """Ghost witness for log-prob: {atom.name}."""')
        lines.append("    n_event = len(dist.event_shape)")
        lines.append("    if n_event > 0:")
        lines.append("        sample_tail = samples.shape[-n_event:]")
        lines.append("        if sample_tail != dist.event_shape:")
        lines.append("            raise ValueError(")
        lines.append(
            '                f"Sample dims {sample_tail} vs event_shape {dist.event_shape}"'
        )
        lines.append("            )")
        lines.append('    return AbstractScalar(dtype="float64", max_val=0.0)')

    elif ct == ConceptType.SAMPLER:
        params = [
            "trace: AbstractMCMCTrace",
            "target: AbstractDistribution",
            "rng: AbstractRNGState",
        ]
        ret = "tuple[AbstractMCMCTrace, AbstractRNGState]"
        lines.append(f"def {witness_name}({', '.join(params)}) -> {ret}:")
        lines.append(f'    """Ghost witness for MCMC sampler: {atom.name}."""')
        lines.append("    if trace.param_dims != target.event_shape:")
        lines.append("        raise ValueError(")
        lines.append(
            '            f"param_dims {trace.param_dims} vs '
            'event_shape {target.event_shape}"'
        )
        lines.append("        )")
        lines.append("    return trace.step(accepted=True), rng.advance(n_draws=1)")

    elif ct == ConceptType.POSTERIOR_UPDATE:
        params = [
            "prior: AbstractDistribution",
            "likelihood: AbstractDistribution",
            "data_shape: tuple[int, ...]",
        ]
        lines.append(
            f"def {witness_name}({', '.join(params)}) -> AbstractDistribution:"
        )
        lines.append(f'    """Ghost witness for posterior update: {atom.name}."""')
        lines.append("    prior.assert_conjugate_to(likelihood)")
        lines.append("    return AbstractDistribution(")
        lines.append("        family=prior.family,")
        lines.append("        event_shape=prior.event_shape,")
        lines.append("        batch_shape=prior.batch_shape,")
        lines.append("        support_lower=prior.support_lower,")
        lines.append("        support_upper=prior.support_upper,")
        lines.append("        is_discrete=prior.is_discrete,")
        lines.append("    )")

    elif ct == ConceptType.CONJUGATE_UPDATE:
        params = [
            "prior: AbstractDistribution",
            "sufficient_stats: AbstractArray",
        ]
        lines.append(
            f"def {witness_name}({', '.join(params)}) -> AbstractDistribution:"
        )
        lines.append(
            f'    """Ghost witness for closed-form conjugate update: {atom.name}."""'
        )
        lines.append(
            "    # Closed-form update: no sampling trace or RNG threading required."
        )
        lines.append("    return AbstractDistribution(")
        lines.append("        family=prior.family,")
        lines.append("        event_shape=prior.event_shape,")
        lines.append("        batch_shape=prior.batch_shape,")
        lines.append("        support_lower=prior.support_lower,")
        lines.append("        support_upper=prior.support_upper,")
        lines.append("        is_discrete=prior.is_discrete,")
        lines.append("    )")

    elif ct == ConceptType.VARIATIONAL_INFERENCE:
        params = [
            "q_dist: AbstractDistribution",
            "p_dist: AbstractDistribution",
            "n_samples: int = 1",
        ]
        lines.append(f"def {witness_name}({', '.join(params)}) -> AbstractScalar:")
        lines.append(f'    """Ghost witness for VI ELBO: {atom.name}."""')
        lines.append("    if q_dist.event_shape != p_dist.event_shape:")
        lines.append("        raise ValueError(")
        lines.append(
            '            f"q event_shape {q_dist.event_shape} vs '
            'p event_shape {p_dist.event_shape}"'
        )
        lines.append("        )")
        lines.append('    return AbstractScalar(dtype="float64")')

    lines.append("")
    return lines


def _generate_message_passing_witness(
    atom: MacroAtomSpec,
    fn_name: str,
    witness_name: str,
) -> list[str]:
    """Generate a memoized witness for a MESSAGE_PASSING atom.

    Routes based on atom name heuristics to produce the appropriate
    witness template for variable-to-factor, factor-to-variable,
    marginal computation, or memoization state nodes.
    """
    lines: list[str] = []
    name_lower = atom.name.lower()

    # Distinguish "Variable to Factor" vs "Factor to Variable" by word order
    var_idx = name_lower.find("variable")
    fac_idx = name_lower.find("factor")
    is_var_to_fac = var_idx >= 0 and fac_idx >= 0 and var_idx < fac_idx
    is_fac_to_var = fac_idx >= 0 and var_idx >= 0 and fac_idx < var_idx

    if is_var_to_fac:
        # Variable-to-Factor message witness
        lines.append(
            f"def {witness_name}(incoming_messages: dict[str, AbstractArray], memo_state: dict[str, AbstractArray]) -> dict[str, AbstractArray]:"
        )
        lines.append('    """Ghost witness for message-passing: Variable to Factor."""')
        lines.append(
            '    _cache_key = ("variable_to_factor", id(incoming_messages), id(memo_state))'
        )
        lines.append("    if _cache_key in _MEMO_CACHE:")
        lines.append("        return _MEMO_CACHE[_cache_key]")
        lines.append(
            "    result = {k: AbstractArray(shape=v.shape, dtype=v.dtype) for k, v in incoming_messages.items()}"
        )
        lines.append("    _MEMO_CACHE[_cache_key] = result")
        lines.append("    return result")

    elif is_fac_to_var:
        # Factor-to-Variable message witness
        lines.append(
            f"def {witness_name}(var_messages: dict[str, AbstractArray], factor_potentials: dict[str, AbstractArray], memo_state: dict[str, AbstractArray]) -> dict[str, AbstractArray]:"
        )
        lines.append('    """Ghost witness for message-passing: Factor to Variable."""')
        lines.append(
            '    _cache_key = ("factor_to_variable", id(var_messages), id(factor_potentials), id(memo_state))'
        )
        lines.append("    if _cache_key in _MEMO_CACHE:")
        lines.append("        return _MEMO_CACHE[_cache_key]")
        lines.append(
            "    result = {k: AbstractArray(shape=v.shape, dtype=v.dtype) for k, v in var_messages.items()}"
        )
        lines.append("    _MEMO_CACHE[_cache_key] = result")
        lines.append("    return result")

    elif "marginal" in name_lower:
        # Marginal computation witness
        lines.append(
            f"def {witness_name}(factor_messages: dict[str, AbstractArray], var_messages: dict[str, AbstractArray]) -> dict[str, AbstractArray]:"
        )
        lines.append(
            '    """Ghost witness for message-passing: Marginal Computation."""'
        )
        lines.append(
            '    _cache_key = ("marginal", id(factor_messages), id(var_messages))'
        )
        lines.append("    if _cache_key in _MEMO_CACHE:")
        lines.append("        return _MEMO_CACHE[_cache_key]")
        lines.append(
            "    result = {k: AbstractArray(shape=v.shape, dtype=v.dtype) for k, v in factor_messages.items()}"
        )
        lines.append("    _MEMO_CACHE[_cache_key] = result")
        lines.append("    return result")

    elif "memo" in name_lower:
        # Memoization state witness
        lines.append(
            f"def {witness_name}(var_messages: dict[str, AbstractArray], factor_messages: dict[str, AbstractArray]) -> tuple[dict[str, AbstractArray], bool]:"
        )
        lines.append('    """Ghost witness for message-passing: Memoization State."""')
        lines.append(
            '    _cache_key = ("memo_state", id(var_messages), id(factor_messages))'
        )
        lines.append("    if _cache_key in _MEMO_CACHE:")
        lines.append("        return _MEMO_CACHE[_cache_key]")
        lines.append(
            "    memo_state = {k: AbstractArray(shape=v.shape, dtype=v.dtype) for k, v in var_messages.items()}"
        )
        lines.append("    converged = False")
        lines.append("    result = (memo_state, converged)")
        lines.append("    _MEMO_CACHE[_cache_key] = result")
        lines.append("    return result")

    else:
        # Generic message-passing witness fallback
        params = []
        for inp in atom.inputs:
            params.append(f"{inp.name}: AbstractArray")
        param_str = ", ".join(params) if params else ""
        lines.append(f"def {witness_name}({param_str}) -> AbstractArray:")
        lines.append(f'    """Ghost witness for message-passing: {atom.name}."""')
        lines.append(f'    _cache_key = ("{fn_name}",)')
        lines.append("    if _cache_key in _MEMO_CACHE:")
        lines.append("        return _MEMO_CACHE[_cache_key]")
        if atom.inputs:
            first = atom.inputs[0].name
            lines.append(
                f"    result = AbstractArray(shape={first}.shape, dtype={first}.dtype)"
            )
        else:
            lines.append('    result = AbstractArray(shape=(), dtype="float32")')
        lines.append("    _MEMO_CACHE[_cache_key] = result")
        lines.append("    return result")

    lines.append("")
    return lines


def generate_ghost_witnesses(
    macro_atoms: list[MacroAtomSpec],
    state_models: list[StateModelSpec] | None = None,
) -> tuple[str, dict[str, str]]:
    """Generate ghost witness functions.

    Returns (source_code, name_mapping) where name_mapping maps
    atom name -> witness function name.

    When *state_models* is non-empty, each witness gains a
    ``state: AbstractSignal`` parameter and returns
    ``tuple[AbstractSignal, AbstractSignal]`` (result, state pass-through).

    Bayesian atoms (concept_type in SAMPLER, LOG_PROB, POSTERIOR_UPDATE,
    VARIATIONAL_INFERENCE, PRIOR_INIT) get specialized witnesses that use
    AbstractDistribution, AbstractRNGState, and AbstractMCMCTrace.
    """
    has_state = bool(state_models)
    has_bayesian = any(a.concept_type in _BAYESIAN_CONCEPT_TYPES for a in macro_atoms)
    has_sampler = any(a.concept_type == ConceptType.SAMPLER for a in macro_atoms)
    has_message_passing = any(
        a.concept_type in _MESSAGE_PASSING_CONCEPT_TYPES for a in macro_atoms
    )

    lines = [
        '"""Auto-generated ghost witness functions for abstract simulation."""',
        "",
        "from __future__ import annotations",
        "",
        "try:",
        "    from ageoa.ghost.abstract import AbstractSignal, AbstractArray, AbstractScalar",
    ]
    if has_bayesian:
        lines.extend(
            [
                "    from ageoa.ghost.abstract import AbstractDistribution",
            ]
        )
    if has_sampler:
        lines.extend(
            [
                "    from ageoa.ghost.abstract import AbstractMCMCTrace",
                "    from ageoa.ghost.abstract import AbstractRNGState",
            ]
        )
    lines.extend(
        [
            "except ImportError:",
            "    pass",
            "",
        ]
    )

    # Memoization cache preamble for message-passing witnesses
    if has_message_passing:
        lines.extend(
            [
                "_MEMO_CACHE: dict = {}",
                "",
                "",
                "def _clear_memo_cache() -> None:",
                '    """Reset the memoization cache between iterations."""',
                "    _MEMO_CACHE.clear()",
                "",
                "",
            ]
        )

    name_map: dict[str, str] = {}

    for atom in macro_atoms:
        if atom.is_opaque:
            continue
        fn_name = _snake_case(atom.name)
        witness_name = f"witness_{fn_name}"
        name_map[atom.name] = witness_name

        # Message-passing atoms get memoized witness templates
        if atom.concept_type in _MESSAGE_PASSING_CONCEPT_TYPES:
            lines.extend(_generate_message_passing_witness(atom, fn_name, witness_name))
            continue

        # Bayesian atoms get specialized witness templates
        if atom.concept_type in _BAYESIAN_CONCEPT_TYPES:
            lines.extend(
                _generate_bayesian_witness(atom, fn_name, witness_name, has_state)
            )
            continue

        # Default DSP/generic witness
        is_dsp = atom.concept_type in {
            ConceptType.SIGNAL_FILTER,
            ConceptType.SIGNAL_TRANSFORM,
            ConceptType.GRAPH_SIGNAL_PROCESSING,
        }
        abstract_type = "AbstractSignal" if is_dsp else "AbstractArray"

        params = []
        for inp in atom.inputs:
            params.append(f"{inp.name}: {abstract_type}")
        if has_state:
            params.append(f"state: {abstract_type}")
        param_str = ", ".join(params) if params else ""

        # Return type
        if has_state:
            if atom.outputs:
                ret_type = f"tuple[{abstract_type}, {abstract_type}]"
            else:
                ret_type = f"tuple[None, {abstract_type}]"
        else:
            if atom.outputs:
                ret_type = abstract_type
            else:
                ret_type = "None"

        lines.append(f"def {witness_name}({param_str}) -> {ret_type}:")
        lines.append(f'    """Ghost witness for {atom.name}."""')

        if atom.inputs and atom.outputs:
            first_input = atom.inputs[0].name
            lines.append(f"    result = {abstract_type}(")
            lines.append(f"        shape={first_input}.shape,")
            lines.append('        dtype="float64",')
            if is_dsp:
                lines.append(
                    f"        sampling_rate=getattr({first_input}, 'sampling_rate', 44100.0),"
                )
                lines.append('        domain="time",')
            lines.append("    )")
            if has_state:
                lines.append("    return result, state")
            else:
                lines.append("    return result")
        else:
            if has_state:
                lines.append("    return None, state")
            else:
                lines.append("    return None")
        lines.append("")

    return "\n".join(lines), name_map


# ---------------------------------------------------------------------------
# Conceptual profile → plain-text summary
# ---------------------------------------------------------------------------


def _profile_to_summary(profile: ConceptualProfile | None) -> str:
    """Convert a ConceptualProfile to a plain-text summary for embedding."""
    if not profile or not profile.abstract_name:
        return ""
    parts = [profile.abstract_name]
    if profile.conceptual_transform:
        parts.append(profile.conceptual_transform)
    if profile.cross_disciplinary_applications:
        parts.append(
            "Applications: " + ", ".join(profile.cross_disciplinary_applications)
        )
    return ". ".join(parts)


# ---------------------------------------------------------------------------
# CDG construction
# ---------------------------------------------------------------------------


def _emit_atom_nodes(
    atom: MacroAtomSpec,
    parent_id: str,
    nodes: list[AlgorithmicNode],
    edges: list[DependencyEdge],
    depth: int,
    seen_node_ids: set[str],
    seen_edge_keys: set[tuple[str, str, str, str, str, str]],
) -> None:
    """Recursively emit CDG nodes for an atom and its children."""
    node_id = _snake_case(atom.name)
    if node_id in seen_node_ids:
        return
    seen_node_ids.add(node_id)

    unique_children: list[MacroAtomSpec] = []
    child_ids: set[str] = set()
    for child in atom.children:
        child_id = _snake_case(child.name)
        if child_id in seen_node_ids or child_id in child_ids:
            continue
        child_ids.add(child_id)
        unique_children.append(child)
    has_children = bool(unique_children)

    node = AlgorithmicNode(
        node_id=node_id,
        parent_id=parent_id,
        name=atom.name,
        description=atom.description,
        concept_type=atom.concept_type,
        inputs=list(atom.inputs),
        outputs=list(atom.outputs),
        status=NodeStatus.DECOMPOSED if has_children else NodeStatus.ATOMIC,
        children=[_snake_case(c.name) for c in unique_children] if has_children else [],
        is_optional=atom.is_optional,
        is_opaque=atom.is_opaque,
        is_external=getattr(atom, "is_external", False),
        type_signature=_build_type_signature(atom),
        conceptual_summary=_profile_to_summary(atom.conceptual_profile),
        depth=depth,
    )
    nodes.append(node)

    for sub_edge in atom.sub_edges:
        emitted = DependencyEdge(
            source_id=_snake_case(sub_edge.source_id),
            target_id=_snake_case(sub_edge.target_id),
            output_name=sub_edge.output_name,
            input_name=sub_edge.input_name,
            source_type=sub_edge.source_type,
            target_type=sub_edge.target_type,
        )
        key = (
            emitted.source_id,
            emitted.target_id,
            emitted.output_name,
            emitted.input_name,
            emitted.source_type,
            emitted.target_type,
        )
        if key in seen_edge_keys:
            continue
        seen_edge_keys.add(key)
        edges.append(emitted)

    for child in unique_children:
        _emit_atom_nodes(
            child,
            node_id,
            nodes,
            edges,
            depth + 1,
            seen_node_ids,
            seen_edge_keys,
        )


def build_cdg_export(plan: ValidatedMacroPlan, class_name: str) -> CDGExport:
    """Build a CDGExport with root DECOMPOSED node + recursive children."""
    unique_top_level: list[MacroAtomSpec] = []
    seen_root_children: set[str] = set()
    for atom in plan.plan.macro_atoms:
        node_id = _snake_case(atom.name)
        if node_id in seen_root_children:
            continue
        seen_root_children.add(node_id)
        unique_top_level.append(atom)

    root_node = AlgorithmicNode(
        node_id=f"{class_name}_root",
        name=class_name,
        description=f"Ingested pipeline from {class_name}",
        concept_type=ConceptType.CUSTOM,
        status=NodeStatus.DECOMPOSED,
        children=[_snake_case(a.name) for a in unique_top_level],
        depth=0,
    )

    all_nodes: list[AlgorithmicNode] = [root_node]
    all_edges: list[DependencyEdge] = []
    seen_node_ids: set[str] = set()
    seen_edge_keys: set[tuple[str, str, str, str, str, str]] = set()

    for atom in unique_top_level:
        _emit_atom_nodes(
            atom,
            root_node.node_id,
            all_nodes,
            all_edges,
            depth=1,
            seen_node_ids=seen_node_ids,
            seen_edge_keys=seen_edge_keys,
        )

    # Build typed edges from plan
    for edge_def in plan.plan.edge_definitions:
        emitted = DependencyEdge(
            source_id=_snake_case(_title_case(edge_def.source_id)),
            target_id=_snake_case(_title_case(edge_def.target_id)),
            output_name=edge_def.output_name,
            input_name=edge_def.input_name,
            source_type=edge_def.source_type,
            target_type=edge_def.target_type,
        )
        key = (
            emitted.source_id,
            emitted.target_id,
            emitted.output_name,
            emitted.input_name,
            emitted.source_type,
            emitted.target_type,
        )
        if key in seen_edge_keys:
            continue
        seen_edge_keys.add(key)
        all_edges.append(emitted)

    return CDGExport(
        nodes=all_nodes,
        edges=all_edges,
        metadata={
            "source": "ingester",
            "class_name": class_name,
        },
    )


def _build_type_signature(atom: MacroAtomSpec) -> str:
    """Build a Python type signature string from IOSpec."""
    inputs = ", ".join(f"{i.name}: {i.type_desc}" for i in atom.inputs)
    if atom.outputs:
        if len(atom.outputs) == 1:
            ret = atom.outputs[0].type_desc
        else:
            ret = "tuple[" + ", ".join(o.type_desc for o in atom.outputs) + "]"
    else:
        ret = "None"
    return f"({inputs}) -> {ret}"


# ---------------------------------------------------------------------------
# Sub-graph construction
# ---------------------------------------------------------------------------


def build_sub_graphs(plan: ValidatedMacroPlan) -> dict[str, CDGExport]:
    """Build zoom-in sub-graphs from sub_atom_refs."""
    sub_graphs: dict[str, CDGExport] = {}

    for atom in plan.plan.macro_atoms:
        # Find sub-atom refs relevant to this macro-atom
        relevant_refs = [
            ref for ref in plan.plan.sub_atom_refs if ref.similarity_score > 0.5
        ]
        if not relevant_refs:
            continue

        node_id = _snake_case(atom.name)
        root = AlgorithmicNode(
            node_id=f"{node_id}_sub_root",
            name=f"{atom.name} (sub-graph)",
            description=f"Zoom-in decomposition of {atom.name}",
            concept_type=atom.concept_type,
            status=NodeStatus.DECOMPOSED,
            children=[_snake_case(r.atom_name) for r in relevant_refs],
            depth=0,
        )

        children = []
        for ref in relevant_refs:
            child = AlgorithmicNode(
                node_id=_snake_case(ref.atom_name),
                parent_id=root.node_id,
                name=ref.atom_name,
                description=f"Existing atom (similarity: {ref.similarity_score:.2f})",
                concept_type=atom.concept_type,
                status=NodeStatus.ATOMIC,
                depth=1,
            )
            children.append(child)

        sub_graphs[node_id] = CDGExport(
            nodes=[root] + children,
            edges=[],
            metadata={"parent_atom": atom.name},
        )

    return sub_graphs


# ---------------------------------------------------------------------------
# Match results
# ---------------------------------------------------------------------------


def build_match_results(cdg: CDGExport, atoms_source: str) -> list[MatchResult]:
    """Build pre-filled MatchResults with verified=True for atomic leaves."""
    results = []
    for node in cdg.nodes:
        if node.status != NodeStatus.ATOMIC:
            continue

        fn_name = _snake_case(node.name)
        decl = Declaration(
            name=fn_name,
            type_signature=node.type_signature,
            docstring=node.description,
            conceptual_summary=node.conceptual_summary,
            source_lib="ingester",
            prover=Prover.PYTHON,
            raw_code="",
        )
        candidate = CandidateMatch(
            declaration=decl,
            score=1.0,
            retrieval_method="ingester",
        )
        vr = VerificationResult(
            candidate=candidate,
            verified=True,
            verification_level=VerificationLevel.TYPE_CHECKED,
        )
        pdg_node = PDGNode(
            predicate_id=node.node_id,
            statement=node.type_signature,
            informal_desc=node.description,
            prover=Prover.PYTHON,
        )
        results.append(
            MatchResult(
                pdg_node=pdg_node,
                verified_match=vr,
                all_candidates=[candidate],
                all_verifications=[vr],
            )
        )

    return results


# ---------------------------------------------------------------------------
# Procedural plan builder (bypasses Phase 2)
# ---------------------------------------------------------------------------


def _title_case(name: str) -> str:
    """Convert a snake_case name like 'remove_baseline' to 'Remove Baseline'."""
    return name.replace("_", " ").title()


_CONCEPT_TYPE_RULES: list[tuple[ConceptType, list[str]]] = [
    (ConceptType.STATE_INIT, ["init", "setup", "create", "reset", "bootstrap", "allocat"]),
    (ConceptType.DATA_ASSEMBLY, [
        "build", "assemble", "construct", "compose", "prepare",
        "materialize", "combine", "merge", "bundle", "package",
    ]),
    (ConceptType.VISUALIZATION, [
        "plot", "render", "draw", "visualiz", "display", "diagnostic", "legend",
    ]),
    (ConceptType.CONDITIONAL_ROUTING, [
        "select", "choose", "route", "dispatch", "gate", "guard",
        "branch", "conditional", "fallback", "default",
    ]),
    (ConceptType.OBSERVABILITY, ["emit", "debug", "log", "trace", "record", "progress"]),
    (ConceptType.DATA_EXTRACTION, [
        "fetch", "load", "read", "parse", "ingest", "decode", "import",
    ]),
]


def _classify_by_name(name: str) -> ConceptType:
    """Classify a function name into a ConceptType using keyword heuristics."""
    lower = name.lower()
    for concept_type, keywords in _CONCEPT_TYPE_RULES:
        if any(kw in lower for kw in keywords):
            return concept_type
    return ConceptType.CUSTOM


def build_procedural_plan(
    dfg: RawDataFlowGraph, pipeline_name: str
) -> ValidatedMacroPlan:
    """Build a ValidatedMacroPlan from procedural SSA edges (no LLM needed).

    Each top-level function becomes one MacroAtomSpec.  Edges come directly
    from ``dfg.inferred_edges`` computed by the SSA visitor.
    """
    macro_atoms: list[MacroAtomSpec] = []
    for mf in dfg.methods:
        inputs = [IOSpec(name=p, type_desc="Any") for p in mf.params]
        outputs = (
            [IOSpec(name="result", type_desc=mf.return_type or "Any")]
            if mf.return_type
            else [IOSpec(name="result", type_desc="Any")]
        )
        if mf.is_external:
            concept_type = ConceptType.EXTERNAL_TOOL
        else:
            concept_type = _classify_by_name(mf.name)
        macro_atoms.append(
            MacroAtomSpec(
                decorators=mf.decorators,
                is_external=mf.is_external,
                concept_type=concept_type,
                name=_title_case(mf.name),
                description=mf.docstring,
                method_names=[mf.name],
                inputs=inputs,
                outputs=outputs,
            )
        )

    plan = ProposedMacroPlan(
        macro_atoms=macro_atoms,
        edge_definitions=list(dfg.inferred_edges),
    )

    return ValidatedMacroPlan(plan=plan, all_attrs_accounted=True)


def _linearize_conjugate_sequence(plan: ValidatedMacroPlan) -> ValidatedMacroPlan:
    """Ensure conjugate updates follow data->update->distribution edges."""
    atoms = plan.plan.macro_atoms
    if not atoms:
        return plan

    edges = list(plan.plan.edge_definitions)
    seen = {
        (
            e.source_id,
            e.target_id,
            e.output_name,
            e.input_name,
            e.source_type,
            e.target_type,
        )
        for e in edges
    }

    def pick_output(atom: MacroAtomSpec) -> tuple[str, str]:
        if atom.outputs:
            pref = next(
                (
                    o
                    for o in atom.outputs
                    if any(
                        h in o.name.lower()
                        for h in (
                            "data",
                            "obs",
                            "sample",
                            "stats",
                            "posterior",
                            "params",
                        )
                    )
                ),
                atom.outputs[0],
            )
            return pref.name, pref.type_desc
        return "result", "Any"

    def pick_input(atom: MacroAtomSpec, out_name: str) -> tuple[str, str]:
        if atom.inputs:
            for inp in atom.inputs:
                if inp.name == out_name:
                    return inp.name, inp.type_desc
            pref = next(
                (
                    i
                    for i in atom.inputs
                    if any(
                        h in i.name.lower()
                        for h in (
                            "data",
                            "obs",
                            "sample",
                            "stats",
                            "posterior",
                            "params",
                        )
                    )
                ),
                atom.inputs[0],
            )
            return pref.name, pref.type_desc
        return out_name, "Any"

    def add_edge(edge: DependencyEdge) -> None:
        key = (
            edge.source_id,
            edge.target_id,
            edge.output_name,
            edge.input_name,
            edge.source_type,
            edge.target_type,
        )
        if key not in seen:
            seen.add(key)
            edges.append(edge)

    for conj in [a for a in atoms if a.concept_type == ConceptType.CONJUGATE_UPDATE]:
        conj_id = _snake_case(conj.name)
        incoming = [e for e in edges if e.target_id == conj_id]
        outgoing = [e for e in edges if e.source_id == conj_id]

        if not incoming:
            data_atom = next(
                (
                    a
                    for a in atoms
                    if a.concept_type != ConceptType.CONJUGATE_UPDATE
                    and _snake_case(a.name) != conj_id
                    and (
                        "data" in a.name.lower()
                        or "ingest" in a.name.lower()
                        or any(
                            h in o.name.lower()
                            for o in a.outputs
                            for h in ("data", "obs", "sample", "stats")
                        )
                    )
                ),
                None,
            )
            if data_atom is not None:
                out_name, out_type = pick_output(data_atom)
                in_name, in_type = pick_input(conj, out_name)
                add_edge(
                    DependencyEdge(
                        source_id=_snake_case(data_atom.name),
                        target_id=conj_id,
                        output_name=out_name,
                        input_name=in_name,
                        source_type=out_type,
                        target_type=in_type,
                    )
                )

        if not outgoing:
            dist_atom = next(
                (
                    a
                    for a in atoms
                    if _snake_case(a.name) != conj_id
                    and (
                        a.concept_type
                        in {ConceptType.PRIOR_DISTRIBUTION, ConceptType.PRIOR_INIT}
                        or "distribution" in a.name.lower()
                        or "posterior" in a.name.lower()
                        or "construct" in a.name.lower()
                    )
                ),
                None,
            )
            if dist_atom is not None:
                out_name, out_type = pick_output(conj)
                in_name, in_type = pick_input(dist_atom, out_name)
                add_edge(
                    DependencyEdge(
                        source_id=conj_id,
                        target_id=_snake_case(dist_atom.name),
                        output_name=out_name,
                        input_name=in_name,
                        source_type=out_type,
                        target_type=in_type,
                    )
                )

    updated = plan.plan.model_copy(update={"edge_definitions": edges})
    return plan.model_copy(update={"plan": updated})


# ---------------------------------------------------------------------------
# Top-level emitter
# ---------------------------------------------------------------------------


def emit_ingestion_bundle(
    plan: ValidatedMacroPlan,
    class_name: str,
    source_file: str = "",
    source_language: str = "python",
) -> IngestionBundle:
    """Assemble all Phase 3 outputs into an IngestionBundle."""
    # Conjugate updates should follow deterministic
    # data->hyperparameter update->distribution construction flow.
    plan = _linearize_conjugate_sequence(plan)

    # Check for opaque atoms
    has_opaque = any(a.is_opaque for a in plan.plan.macro_atoms)

    # Generate witnesses first (need name mapping for atoms)
    witness_source, witness_names = generate_ghost_witnesses(
        plan.plan.macro_atoms,
        state_models=plan.plan.state_models,
    )

    if has_opaque:
        # Append opaque witness stubs (fallback, no LLM)
        opaque_lines = [
            "",
            "# Opaque DL boundaries",
            "",
        ]
        for atom in plan.plan.macro_atoms:
            if atom.is_opaque:
                opaque_lines.append(_opaque_witness_fallback(atom))
                opaque_lines.append("")
                fn_name = _snake_case(atom.name)
                witness_names[atom.name] = f"witness_{fn_name}"
        witness_source += "\n".join(opaque_lines)

    # Generate state models
    state_model_source = generate_state_models(plan.plan.state_models)

    # Generate atom wrappers — stateful if state models exist
    if plan.plan.state_models:
        atoms_source = generate_stateful_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            class_name,
            witness_names,
            source_file=source_file,
            source_language=source_language,
            plan=plan,
        )
    else:
        atoms_source = generate_atom_wrappers(
            plan.plan.macro_atoms,
            plan.plan.state_models,
            witness_names,
            source_language=source_language,
            class_name=class_name,
            source_file=source_file,
            plan=plan,
        )

    # Append FFI binding stubs for non-Python sources
    if source_language != "python":
        ffi_source = generate_ffi_bindings(plan.plan.macro_atoms, source_language)
        atoms_source = atoms_source + "\n\n" + ffi_source

    # Build CDG
    cdg = build_cdg_export(plan, class_name)

    # Build sub-graphs
    sub_graphs = build_sub_graphs(plan)

    # Build match results
    match_results = build_match_results(cdg, atoms_source)

    return IngestionBundle(
        cdg=cdg,
        sub_graphs=sub_graphs,
        generated_atoms=atoms_source,
        generated_state_models=state_model_source,
        generated_witnesses=witness_source,
        match_results=match_results,
    )

"""Phase 2: LangGraph semantic chunking sub-graph.

Groups methods into macro-atoms, hoists cross-window state, searches
for existing sub-atoms, and validates coverage via a critic loop.
"""

from __future__ import annotations

import asyncio
import json
import logging

from sciona.json_utils import extract_json
from dataclasses import dataclass
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph
from typing_extensions import TypedDict

from sciona.architect.catalog import PrimitiveCatalog
from sciona.architect.deterministic_decompose import _infer_concept_type
from sciona.hunter.llm import LLMClient
from sciona.architect.models import ConceptType, IOSpec
from sciona.ingester.monitor import IngestMonitor
from sciona.ingester.models import (
    ConceptualProfile,
    DependencyEdge,
    IngestIRPlan,
    MacroAtomSpec,
    MethodFact,
    MethodBinding,
    OperationEdge,
    OperationSpec,
    OutputBindingSpec,
    ProposedMacroPlan,
    RawDataFlowGraph,
    StateEffectSpec,
    StateSlotSpec,
    StateModelSpec,
    SubAtomRef,
    ValidatedMacroPlan,
)
from sciona.ingester.prompts import (
    CONCEPTUAL_ABSTRACT_SYSTEM,
    CONCEPTUAL_ABSTRACT_USER,
    DECOMPOSE_ATOM_SYSTEM,
    DECOMPOSE_ATOM_USER,
    HOIST_STATE_SYSTEM,
    HOIST_STATE_USER,
    SEMANTIC_CHUNK_SYSTEM,
    SEMANTIC_CHUNK_USER,
)
from sciona.llm_router import (
    INGESTER_ABSTRACT,
    INGESTER_CHUNK,
    INGESTER_DECOMPOSE,
    INGESTER_HOIST_STATE,
    select_llm,
)
from sciona.shared_context import (
    SharedContextMetrics,
    SharedContextStore,
    format_context_block,
)
from sciona.protocols import SemanticIndex

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_IGNORED_METHOD_NAMES = {"__init__", "__repr__", "__str__"}


# ---------------------------------------------------------------------------
# State & deps
# ---------------------------------------------------------------------------


class ChunkerState(TypedDict):
    raw_dfg: RawDataFlowGraph
    proposed_plan: ProposedMacroPlan
    validated_plan: ValidatedMacroPlan
    critique_passed: bool
    critique_reason: str
    retry_count: int
    missing_attrs: list[str]
    done: bool


@dataclass
class ChunkerDeps:
    llm: LLMClient
    faiss_index: SemanticIndex | None = None
    max_depth: int = 1
    line_threshold: int = 30
    monitor: IngestMonitor | None = None
    shared_context: SharedContextStore | None = None
    shared_context_metrics: SharedContextMetrics | None = None
    context_namespace: str = ""
    context_budget_chars: int = 900
    parallelism: int = 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_monitor(config: RunnableConfig) -> IngestMonitor | None:
    deps = config.get("configurable", {}).get("deps")
    if deps is None:
        return None
    return getattr(deps, "monitor", None)


def _build_method_summaries(dfg: RawDataFlowGraph) -> str:
    lines = []
    for mf in dfg.methods:
        reads = ", ".join(mf.reads) if mf.reads else "(none)"
        writes = ", ".join(mf.writes) if mf.writes else "(none)"
        calls = ", ".join(mf.calls) if mf.calls else "(none)"
        lines.append(
            f"- {mf.name}({', '.join(mf.params)})"
            f"\n  reads: {reads}"
            f"\n  writes: {writes}"
            f"\n  calls: {calls}"
        )
    return "\n".join(lines)


def _build_attr_graph(dfg: RawDataFlowGraph) -> str:
    lines = []
    for attr, accesses in sorted(dfg.all_attributes.items()):
        lines.append(f"  {attr}: {accesses}")
    return "\n".join(lines)


def _build_config_branches(dfg: RawDataFlowGraph) -> str:
    if not dfg.config_branches:
        return "(none)"
    lines = []
    for cb in dfg.config_branches:
        lines.append(
            f"  if self.options.{cb.config_attr} in {cb.method} "
            f"(lines {cb.lines[0]}-{cb.lines[1]})"
        )
    return "\n".join(lines)


def _op_id(text: str) -> str:
    return text.lower().replace(" ", "_").replace("-", "_")


def _plan_state_model_name(subject_name: str) -> str:
    return f"{subject_name}State"


def _operation_role(methods: list[MethodFact]) -> str:
    roles = [method.semantic_role for method in methods if method.semantic_role]
    if not roles:
        return "unknown"
    if "constructor" in roles:
        return "constructor"
    if "fit_or_update" in roles:
        return "state_transition"
    if "predict_or_transform" in roles:
        primary_names = " ".join(method.name.lower() for method in methods)
        if "transform" in primary_names:
            return "transform"
        return "predict"
    if "score_or_evaluate" in roles:
        return "score"
    if "query_or_metadata" in roles:
        names = " ".join(method.name.lower() for method in methods)
        if any(token in names for token in ("metadata", "tag", "routing")):
            return "metadata"
        return "query"
    if all(role == "helper" for role in roles):
        return "helper"
    return "unknown"


def _state_kind_for_attr(dfg: RawDataFlowGraph, attr_name: str) -> str:
    if attr_name in set(dfg.config_attributes):
        return "config"
    if attr_name in set(dfg.fitted_attributes):
        return "fitted"
    if attr_name in set(dfg.derived_attributes):
        return "derived"
    return "transient"


def _slot_type_hint(
    attr_name: str,
    legacy_plan: ProposedMacroPlan | None,
) -> str:
    if legacy_plan is None:
        return "Any"
    for state_model in legacy_plan.state_models:
        for field_name, field_type in state_model.fields:
            if field_name == attr_name and field_type:
                return field_type
    for atom in legacy_plan.macro_atoms:
        for io in [*atom.inputs, *atom.outputs]:
            if io.name == attr_name and io.type_desc:
                return io.type_desc
    return "Any"


def _build_state_slots(
    dfg: RawDataFlowGraph,
    legacy_plan: ProposedMacroPlan | None,
) -> list[StateSlotSpec]:
    facts_by_attr = {fact.attr_name: fact for fact in dfg.attribute_facts}
    attr_names = set(facts_by_attr) | set(dfg.all_attributes)
    slots: list[StateSlotSpec] = []
    for attr_name in sorted(attr_names):
        fact = facts_by_attr.get(attr_name)
        read_methods: list[str]
        write_methods: list[str]
        provenances = list(fact.provenances) if fact else []
        if fact is not None:
            read_methods = sorted(fact.read_methods)
            write_methods = sorted(fact.write_methods)
        else:
            read_methods = []
            write_methods = []
            for access in dfg.all_attributes.get(attr_name, []):
                prefix, _, method_name = access.partition(":")
                if not method_name:
                    continue
                if prefix == "read":
                    read_methods.append(method_name)
                elif prefix == "write":
                    write_methods.append(method_name)
        slots.append(
            StateSlotSpec(
                slot_name=attr_name,
                state_kind=_state_kind_for_attr(dfg, attr_name),
                type_desc=_slot_type_hint(attr_name, legacy_plan),
                required_before=sorted(read_methods),
                written_by=sorted(write_methods),
                read_by=sorted(read_methods),
                source_attr=attr_name,
                provenance=provenances,
            )
        )
    return slots


def _method_binding(method: MethodFact) -> MethodBinding:
    kinds = [param.kind for param in method.signature]
    call_style = ",".join(kinds) if kinds else ""
    return MethodBinding(
        method_name=method.name,
        signature=list(method.signature),
        call_style=call_style,
        return_behavior=list(method.return_facts),
        requires_instance_state=bool(method.reads or method.writes),
        provenance=list(method.provenance),
    )


def _default_direct_inputs(methods: list[MethodFact]) -> list[IOSpec]:
    seen: set[str] = set()
    inputs: list[IOSpec] = []
    for method in methods:
        for param in method.signature:
            if param.name in seen:
                continue
            seen.add(param.name)
            inputs.append(
                IOSpec(
                    name=param.name,
                    type_desc=param.annotation or "Any",
                )
            )
    return inputs


def _binding_from_return_fact(
    method: MethodFact,
    fact,
    output_name: str,
    type_desc: str,
    tuple_index: int | None = None,
) -> OutputBindingSpec:
    method_name = method.name.lower()
    is_metadata_method = any(token in method_name for token in ("metadata", "routing", "tag"))
    if fact.kind == "attribute":
        binding_kind = "attribute_read"
    elif fact.kind == "call_result":
        binding_kind = "metadata_object" if is_metadata_method else "return_value"
    elif fact.kind == "tuple":
        binding_kind = "tuple_element"
    elif fact.kind == "self":
        binding_kind = "self_return"
    elif fact.kind == "constant":
        binding_kind = "metadata_object" if is_metadata_method else "constant"
    else:
        binding_kind = "unknown"
    source_attr = fact.referenced_attrs[0] if fact.referenced_attrs else ""
    return OutputBindingSpec(
        output_name=output_name,
        type_desc=type_desc or "Any",
        binding_kind=binding_kind,
        source_method=method.name,
        source_attr=source_attr,
        tuple_index=tuple_index,
        provenance=[fact.provenance],
    )


def _match_output_name_to_attr(output_name: str, attrs: set[str]) -> str:
    if output_name in attrs:
        return output_name
    lowered = output_name.lower()
    candidates = [
        attr
        for attr in attrs
        if attr.lower() in lowered or lowered in attr.lower()
    ]
    if len(candidates) == 1:
        return candidates[0]
    return ""


def _infer_output_bindings(
    methods: list[MethodFact],
    legacy_outputs: list[IOSpec],
) -> list[OutputBindingSpec]:
    if not methods:
        return []

    return_pairs: list[tuple[MethodFact, Any]] = []
    for method in methods:
        for fact in method.return_facts:
            if fact.kind != "none":
                return_pairs.append((method, fact))

    writes = {attr for method in methods for attr in method.writes}
    bindings: list[OutputBindingSpec] = []

    if legacy_outputs:
        for legacy_output in legacy_outputs:
            matched_attr = _match_output_name_to_attr(legacy_output.name, writes)
            attribute_fact = next(
                (
                    (method, fact)
                    for method, fact in return_pairs
                    if (
                        legacy_output.name in fact.referenced_attrs
                        or (
                            matched_attr
                            and matched_attr in fact.referenced_attrs
                        )
                    )
                ),
                None,
            )
            if matched_attr or attribute_fact is not None:
                method_name = methods[-1].name if attribute_fact is None else attribute_fact[0].name
                provenance = [] if attribute_fact is None else [attribute_fact[1].provenance]
                bindings.append(
                    OutputBindingSpec(
                        output_name=legacy_output.name,
                        type_desc=legacy_output.type_desc or "Any",
                        binding_kind="attribute_read",
                        source_method=method_name,
                        source_attr=matched_attr or legacy_output.name,
                        provenance=provenance,
                    )
                )
                continue

            tuple_outputs = [pair for pair in return_pairs if pair[1].kind == "tuple"]
            if tuple_outputs:
                method, fact = tuple_outputs[0]
                index = legacy_outputs.index(legacy_output)
                source_attr = (
                    fact.referenced_attrs[index]
                    if index < len(fact.referenced_attrs)
                    else legacy_output.name
                )
                bindings.append(
                    OutputBindingSpec(
                        output_name=legacy_output.name,
                        type_desc=legacy_output.type_desc or "Any",
                        binding_kind="tuple_element",
                        source_method=method.name,
                        source_attr=source_attr,
                        tuple_index=index,
                        provenance=[fact.provenance],
                    )
                )
                continue

            return_pair = next(
                (
                    (method, fact)
                    for method, fact in return_pairs
                    if fact.kind not in {"self", "none"}
                ),
                None,
            )
            if return_pair is not None:
                method, fact = return_pair
                bindings.append(
                    _binding_from_return_fact(
                        method,
                        fact,
                        legacy_output.name,
                        legacy_output.type_desc,
                    )
                )
                continue

            if len(writes) == 1:
                inferred_attr = next(iter(writes))
                bindings.append(
                    OutputBindingSpec(
                        output_name=legacy_output.name,
                        type_desc=legacy_output.type_desc or "Any",
                        binding_kind="attribute_read",
                        source_method=methods[-1].name,
                        source_attr=inferred_attr,
                    )
                )
                continue

            bindings.append(
                OutputBindingSpec(
                    output_name=legacy_output.name,
                    type_desc=legacy_output.type_desc or "Any",
                    binding_kind="unknown",
                    source_method=methods[-1].name,
                )
            )
        return bindings

    for method, fact in return_pairs:
        if fact.kind == "self":
            continue
        if fact.kind == "attribute":
            attr_name = fact.referenced_attrs[0] if fact.referenced_attrs else "result"
            bindings.append(
                _binding_from_return_fact(method, fact, attr_name, method.return_type or "Any")
            )
            continue
        if fact.kind == "tuple":
            if fact.referenced_attrs:
                for index, attr_name in enumerate(fact.referenced_attrs):
                    bindings.append(
                        OutputBindingSpec(
                            output_name=attr_name,
                            type_desc="Any",
                            binding_kind="tuple_element",
                            source_method=method.name,
                            source_attr=attr_name,
                            tuple_index=index,
                            provenance=[fact.provenance],
                        )
                    )
            else:
                bindings.append(
                    OutputBindingSpec(
                        output_name="result",
                        type_desc=method.return_type or "Any",
                        binding_kind="tuple_element",
                        source_method=method.name,
                        provenance=[fact.provenance],
                    )
                )
            continue
        bindings.append(
            _binding_from_return_fact(method, fact, "result", method.return_type or "Any")
        )

    if bindings:
        return bindings

    for attr_name in sorted(writes):
        bindings.append(
            OutputBindingSpec(
                output_name=attr_name,
                type_desc="Any",
                binding_kind="attribute_read",
                source_method=methods[-1].name,
                source_attr=attr_name,
            )
        )
    return bindings


def _infer_state_effects(
    methods: list[MethodFact],
    state_slot_names: set[str],
) -> list[StateEffectSpec]:
    effects: list[StateEffectSpec] = []
    for method in methods:
        seen: set[str] = set()
        for attr_name in [*method.reads, *method.writes]:
            if attr_name not in state_slot_names or attr_name in seen:
                continue
            seen.add(attr_name)
            if attr_name in method.writes:
                effect_kind = "initialize" if method.semantic_role == "constructor" else "update"
            else:
                effect_kind = "read_only"
            effects.append(
                StateEffectSpec(
                    slot_name=attr_name,
                    effect_kind=effect_kind,
                    source_method=method.name,
                    provenance=list(method.provenance),
                )
            )
    return effects


def _default_operations(dfg: RawDataFlowGraph) -> list[MacroAtomSpec]:
    if dfg.is_opaque and dfg.methods:
        method = dfg.methods[0]
        return [
            MacroAtomSpec(
                name=dfg.class_name,
                description=method.docstring or dfg.class_name,
                method_names=[method.name],
                inputs=[IOSpec(name=param, type_desc="Any") for param in method.params],
                outputs=[IOSpec(name="output", type_desc=method.return_type or "Any")],
                concept_type=ConceptType.NEURAL_NETWORK,
                is_opaque=True,
            )
        ]
    atoms: list[MacroAtomSpec] = []
    for method in dfg.methods:
        if not _is_public_method_name(method.name):
            continue
        atoms.append(
            MacroAtomSpec(
                name=_atom_name_for_method(method.name),
                description=method.docstring or method.name.replace("_", " "),
                method_names=[method.name],
                inputs=_default_direct_inputs([method]),
                outputs=_outputs_for_method(method),
                concept_type=_infer_method_concept(method),
            )
        )
    return atoms


def _build_ingest_ir(
    dfg: RawDataFlowGraph,
    legacy_plan: ProposedMacroPlan | None = None,
) -> IngestIRPlan:
    by_name = {method.name: method for method in dfg.methods}
    state_slots = _build_state_slots(dfg, legacy_plan)
    state_slot_names = {slot.slot_name for slot in state_slots}
    seed_atoms = _default_operations(dfg) if legacy_plan is None else list(legacy_plan.macro_atoms)

    operations: list[OperationSpec] = []
    artifacts: list[OutputBindingSpec] = []
    for atom in seed_atoms:
        methods = [by_name[name] for name in atom.method_names if name in by_name]
        if not methods:
            continue
        outputs = _infer_output_bindings(methods, atom.outputs)
        operation = OperationSpec(
            operation_id=_op_id(atom.name),
            display_name=atom.name,
            role=_operation_role(methods),
            method_bindings=[_method_binding(method) for method in methods],
            direct_inputs=list(atom.inputs) if atom.inputs else _default_direct_inputs(methods),
            required_state_slots=sorted(
                {
                    attr
                    for method in methods
                    for attr in method.reads
                    if attr in state_slot_names
                }
            ),
            emitted_outputs=outputs,
            state_effects=_infer_state_effects(methods, state_slot_names),
            concept_type=atom.concept_type,
            is_optional=atom.is_optional,
            is_opaque=atom.is_opaque,
            is_external=atom.is_external,
            provenance=[prov for method in methods for prov in method.provenance],
        )
        operations.append(operation)
        artifacts.extend(outputs)

    edges: list[OperationEdge] = []
    seen_edges: set[tuple[str, str, str, str]] = set()
    if legacy_plan is not None:
        for edge in legacy_plan.edge_definitions:
            edge_kind = (
                "state"
                if edge.output_name in state_slot_names or edge.input_name in state_slot_names
                else "data"
            )
            key = (edge.source_id, edge.target_id, edge_kind, edge.output_name or edge.input_name)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            edges.append(
                OperationEdge(
                    source_operation_id=edge.source_id,
                    target_operation_id=edge.target_id,
                    edge_kind=edge_kind,
                    artifact_or_slot_name=edge.output_name or edge.input_name,
                )
            )

    for slot in state_slots:
        for writer in slot.written_by:
            source_operation_id = None
            for operation in operations:
                if writer in {binding.method_name for binding in operation.method_bindings}:
                    source_operation_id = operation.operation_id
                    break
            if source_operation_id is None:
                continue
            for reader in slot.read_by:
                target_operation_id = None
                for operation in operations:
                    if reader in {binding.method_name for binding in operation.method_bindings}:
                        target_operation_id = operation.operation_id
                        break
                if target_operation_id is None or target_operation_id == source_operation_id:
                    continue
                key = (source_operation_id, target_operation_id, "state", slot.slot_name)
                if key in seen_edges:
                    continue
                seen_edges.add(key)
                edges.append(
                    OperationEdge(
                        source_operation_id=source_operation_id,
                        target_operation_id=target_operation_id,
                        edge_kind="state",
                        artifact_or_slot_name=slot.slot_name,
                        provenance=list(slot.provenance),
                    )
                )

    return IngestIRPlan(
        subject_name=dfg.class_name,
        source_language=dfg.source_language,
        operations=operations,
        state_slots=state_slots,
        artifacts=artifacts,
        edges=edges,
        unknowns=list(dfg.semantic_unknowns),
    )


def _legacy_outputs_from_operation(operation: OperationSpec) -> list[IOSpec]:
    outputs: list[IOSpec] = []
    for binding in operation.emitted_outputs:
        if binding.binding_kind == "self_return":
            continue
        outputs.append(IOSpec(name=binding.output_name, type_desc=binding.type_desc))
    return outputs


def _legacy_state_models_from_ir(
    ir: IngestIRPlan,
    existing_state_models: list[StateModelSpec],
) -> list[StateModelSpec]:
    if existing_state_models:
        return existing_state_models
    slots = [
        slot
        for slot in ir.state_slots
        if slot.state_kind in {"fitted", "derived", "stochastic"}
    ]
    if not slots:
        return []
    return [
        StateModelSpec(
            model_name=_plan_state_model_name(ir.subject_name),
            fields=[(slot.slot_name, slot.type_desc or "Any") for slot in slots],
            source_attrs=[slot.slot_name for slot in slots],
            docstring=f"Legacy adapter state model for {ir.subject_name}.",
        )
    ]


def _legacy_edges_from_ir(ir: IngestIRPlan) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    for edge in ir.edges:
        if edge.edge_kind not in {"data", "state"}:
            continue
        edges.append(
            DependencyEdge(
                source_id=edge.source_operation_id,
                target_id=edge.target_operation_id,
                output_name=edge.artifact_or_slot_name,
                input_name=edge.artifact_or_slot_name,
                source_type="Any",
                target_type="Any",
            )
        )
    return edges


def _adapt_ir_to_legacy_plan(
    ir: IngestIRPlan,
    existing_plan: ProposedMacroPlan | None = None,
) -> ProposedMacroPlan:
    existing_plan = existing_plan or ProposedMacroPlan()
    by_op_id = {_op_id(atom.name): atom for atom in existing_plan.macro_atoms}
    macro_atoms: list[MacroAtomSpec] = []
    for operation in ir.operations:
        seed = by_op_id.get(operation.operation_id)
        macro_atoms.append(
            MacroAtomSpec(
                name=seed.name if seed is not None else operation.display_name,
                description=seed.description if seed is not None else operation.display_name,
                method_names=[binding.method_name for binding in operation.method_bindings],
                inputs=list(seed.inputs) if seed is not None and seed.inputs else list(operation.direct_inputs),
                outputs=(
                    list(seed.outputs)
                    if seed is not None and seed.outputs
                    else _legacy_outputs_from_operation(operation)
                ),
                config_params=list(seed.config_params) if seed is not None else [],
                concept_type=seed.concept_type if seed is not None else operation.concept_type,
                decorators=list(seed.decorators) if seed is not None else [],
                is_optional=seed.is_optional if seed is not None else operation.is_optional,
                is_opaque=seed.is_opaque if seed is not None else operation.is_opaque,
                is_external=seed.is_external if seed is not None else operation.is_external,
                is_stochastic=seed.is_stochastic if seed is not None else False,
                requires_rng_key=seed.requires_rng_key if seed is not None else False,
                requires_autodiff=seed.requires_autodiff if seed is not None else False,
                autodiff_backend=seed.autodiff_backend if seed is not None else "",
                conceptual_profile=seed.conceptual_profile if seed is not None else None,
                children=list(seed.children) if seed is not None else [],
                sub_edges=list(seed.sub_edges) if seed is not None else [],
                depth=seed.depth if seed is not None else 0,
                source_lines=seed.source_lines if seed is not None else 0,
            )
        )

    return ProposedMacroPlan(
        macro_atoms=macro_atoms,
        state_models=_legacy_state_models_from_ir(ir, existing_plan.state_models),
        sub_atom_refs=list(existing_plan.sub_atom_refs),
        edge_definitions=(
            list(existing_plan.edge_definitions)
            if existing_plan.edge_definitions
            else _legacy_edges_from_ir(ir)
        ),
        canonical_ir=ir,
    )


def _attach_canonical_ir(
    dfg: RawDataFlowGraph,
    legacy_plan: ProposedMacroPlan,
) -> ProposedMacroPlan:
    ir = _build_ingest_ir(dfg, legacy_plan)
    return _adapt_ir_to_legacy_plan(ir, existing_plan=legacy_plan)


def _validate_canonical_ir(
    dfg: RawDataFlowGraph,
    ir: IngestIRPlan,
) -> tuple[bool, str, list[str]]:
    issues: list[str] = []
    slot_names = {slot.slot_name for slot in ir.state_slots}
    covered_methods = {
        binding.method_name for operation in ir.operations for binding in operation.method_bindings
    }
    covered_attrs = {
        slot.source_attr
        for slot in ir.state_slots
        if slot.source_attr
        and (
            set(slot.read_by).intersection(covered_methods)
            or set(slot.written_by).intersection(covered_methods)
        )
    }
    method_to_role = {
        binding.method_name: operation.role
        for operation in ir.operations
        for binding in operation.method_bindings
    }

    for operation in ir.operations:
        if not operation.method_bindings:
            issues.append(f"{operation.display_name}: missing method bindings")
        for required_slot in operation.required_state_slots:
            if required_slot not in slot_names:
                issues.append(f"{operation.display_name}: unknown required state slot {required_slot}")
        if operation.role in {"query", "metadata", "score"}:
            mutating = [
                effect.slot_name
                for effect in operation.state_effects
                if effect.effect_kind in {"initialize", "update", "clear"}
            ]
            if mutating:
                issues.append(
                    f"{operation.display_name}: {operation.role} operation mutates state {sorted(mutating)}"
                )
        for binding in operation.emitted_outputs:
            if not binding.source_method:
                issues.append(f"{operation.display_name}: output {binding.output_name} missing source method")
            if binding.binding_kind == "unknown":
                issues.append(f"{operation.display_name}: output {binding.output_name} has unknown binding")
            if binding.source_attr:
                covered_attrs.add(binding.source_attr)

    for slot in ir.state_slots:
        if slot.state_kind == "transient" and slot.slot_name in set(dfg.fitted_attributes):
            issues.append(f"{slot.slot_name}: fitted attribute downgraded to transient")
        if slot.state_kind == "config" and slot.slot_name not in set(dfg.config_attributes):
            attr_roles = {
                method_to_role.get(method_name, "")
                for method_name in [*slot.read_by, *slot.written_by]
            }
            if "constructor" not in attr_roles:
                issues.append(f"{slot.slot_name}: config slot lacks constructor provenance")

    missing_attrs = sorted(set(dfg.all_attributes) - covered_attrs)
    if missing_attrs:
        issues.append(f"Missing attributes: {missing_attrs}")

    report = "All canonical IR checks passed." if not issues else "; ".join(issues)
    return not issues, report, missing_attrs


async def _search_context(
    deps: ChunkerDeps,
    *,
    channel: str,
    query: str,
    limit: int = 3,
) -> str:
    store = deps.shared_context
    ns = deps.context_namespace
    if store is None or not ns:
        return ""
    try:
        records = await store.search(f"{ns}/{channel}", query, limit=limit)
        return format_context_block(
            "Shared Context",
            records,
            max_chars=deps.context_budget_chars,
            metrics=deps.shared_context_metrics,
        )
    except Exception:
        return ""


async def _put_context(
    deps: ChunkerDeps,
    *,
    channel: str,
    text: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    store = deps.shared_context
    ns = deps.context_namespace
    if store is None or not ns:
        return
    try:
        await store.put(f"{ns}/{channel}", text, metadata=metadata)
    except Exception:
        return


def _compute_state_edges(
    dfg: RawDataFlowGraph,
    plan: ProposedMacroPlan,
) -> list[DependencyEdge]:
    """Compute deterministic state-typed edges from method read/write sets.

    For each state attribute, creates edges from every writer atom to every
    reader atom (deduplicated, self-loops excluded).  The edge type is the
    state model name (e.g. ``RollingAveragerState``).
    """
    if not plan.state_models:
        return []

    state_model = plan.state_models[0]
    state_model_name = state_model.model_name
    state_attrs = set(state_model.source_attrs)

    # Map method name -> atom snake_case id
    method_to_atom: dict[str, str] = {}
    for atom in plan.macro_atoms:
        atom_id = atom.name.lower().replace(" ", "_").replace("-", "_")
        for mname in atom.method_names:
            method_to_atom[mname] = atom_id

    # For each method, check if it reads/writes state attrs
    writers: dict[str, set[str]] = {}  # attr -> set of atom_ids
    readers: dict[str, set[str]] = {}  # attr -> set of atom_ids

    for mf in dfg.methods:
        if mf.name == "__init__":
            continue
        atom_id = method_to_atom.get(mf.name)
        if atom_id is None:
            continue
        for attr in mf.writes:
            if attr in state_attrs:
                writers.setdefault(attr, set()).add(atom_id)
        for attr in mf.reads:
            if attr in state_attrs:
                readers.setdefault(attr, set()).add(atom_id)

    # Build edges: writer -> reader for each attr
    seen: set[tuple[str, str]] = set()
    edges: list[DependencyEdge] = []

    for attr in state_attrs:
        for w in writers.get(attr, set()):
            for r in readers.get(attr, set()):
                if w == r:
                    continue
                key = (w, r)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    DependencyEdge(
                        source_id=w,
                        target_id=r,
                        output_name=attr,
                        input_name=attr,
                        source_type=state_model_name,
                        target_type=state_model_name,
                    )
                )

    return edges


def _is_public_method_name(name: str) -> bool:
    return not name.startswith("_") and name not in _IGNORED_METHOD_NAMES


def _method_line_count(method: MethodFact) -> int:
    source = (method.source_code or "").strip()
    return len(source.splitlines()) if source else 0


def _is_simple_class(dfg: RawDataFlowGraph) -> bool:
    public_methods = [method for method in dfg.methods if _is_public_method_name(method.name)]
    if not public_methods:
        return False
    if any(_method_line_count(method) > 30 for method in public_methods):
        return False
    internal_dispatch = dfg.internal_call_graph or {}
    public_names = {method.name for method in public_methods}
    for method in public_methods:
        direct_calls = set(method.calls) | set(internal_dispatch.get(method.name, []))
        if direct_calls & public_names:
            return False
    # TODO: relax to `len(...) > 3` once simple-class path handles stateful classes
    if dfg.cross_window_attrs:
        return False
    if dfg.config_branches:
        return False
    base_classes = [base for base in dfg.opaque_base_classes if base and base != "object"]
    if base_classes:
        return False
    touched_attrs = {attr for method in public_methods for attr in (*method.reads, *method.writes)}
    if set(dfg.all_attributes) - touched_attrs:
        return False
    return True


def _dedupe_ios(specs: list[IOSpec]) -> list[IOSpec]:
    seen: set[str] = set()
    deduped: list[IOSpec] = []
    for spec in specs:
        if spec.name in seen:
            continue
        seen.add(spec.name)
        deduped.append(spec)
    return deduped


def _atom_name_for_method(method_name: str) -> str:
    return method_name.replace("_", " ").strip().title()


def _infer_method_concept(method: MethodFact) -> ConceptType:
    catalog = PrimitiveCatalog()
    return _infer_concept_type(
        {
            "name": _atom_name_for_method(method.name),
            "description": method.docstring or method.name.replace("_", " "),
        },
        parent_type=ConceptType.CUSTOM,
        catalog=catalog,
    )


def _outputs_for_method(method: MethodFact) -> list[IOSpec]:
    outputs: list[IOSpec] = []
    return_type = (method.return_type or "").strip()
    if return_type and return_type.lower() != "none":
        outputs.append(IOSpec(name="result", type_desc=return_type))
    outputs.extend(IOSpec(name=attr, type_desc="Any") for attr in method.writes)
    return _dedupe_ios(outputs)


def _inputs_for_method(method: MethodFact) -> list[IOSpec]:
    inputs = [
        IOSpec(name=param, type_desc="Any")
        for param in method.params
        if param and param != "self"
    ]
    inputs.extend(IOSpec(name=attr, type_desc="Any") for attr in method.reads)
    return _dedupe_ios(inputs)


def _deterministic_chunk_edges(atoms: list[MacroAtomSpec]) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    seen: set[tuple[str, str, str, str]] = set()
    for idx, atom in enumerate(atoms):
        source_outputs = {output.name: output.type_desc for output in atom.outputs}
        for next_idx in range(idx + 1, len(atoms)):
            next_atom = atoms[next_idx]
            target_inputs = {input_.name: input_.type_desc for input_ in next_atom.inputs}
            shared_names = sorted(set(source_outputs) & set(target_inputs))
            if shared_names:
                for name in shared_names:
                    key = (atom.name, next_atom.name, name, name)
                    if key in seen:
                        continue
                    seen.add(key)
                    edges.append(
                        DependencyEdge(
                            source_id=atom.name,
                            target_id=next_atom.name,
                            output_name=name,
                            input_name=name,
                            source_type=source_outputs[name] or "Any",
                            target_type=target_inputs[name] or "Any",
                        )
                    )
                continue

            if next_idx != idx + 1:
                continue
            if atom.outputs and next_atom.inputs:
                src = atom.outputs[0]
                tgt = next_atom.inputs[0]
                key = (atom.name, next_atom.name, src.name, tgt.name)
                if key in seen:
                    continue
                seen.add(key)
                edges.append(
                    DependencyEdge(
                        source_id=atom.name,
                        target_id=next_atom.name,
                        output_name=src.name,
                        input_name=tgt.name,
                        source_type=src.type_desc or "Any",
                        target_type=tgt.type_desc or "Any",
                    )
                )
    return edges


def _chunk_by_method(dfg: RawDataFlowGraph) -> ProposedMacroPlan:
    public_methods = [method for method in dfg.methods if _is_public_method_name(method.name)]
    atoms = [
        MacroAtomSpec(
            name=_atom_name_for_method(method.name),
            description=method.docstring or f"Execute {method.name.replace('_', ' ')}.",
            method_names=[method.name],
            inputs=_inputs_for_method(method),
            outputs=_outputs_for_method(method),
            concept_type=_infer_method_concept(method),
        )
        for method in public_methods
    ]
    return ProposedMacroPlan(
        macro_atoms=atoms,
        edge_definitions=_deterministic_chunk_edges(atoms),
    )


def _parse_macro_atoms(raw: dict) -> list[MacroAtomSpec]:
    if not isinstance(raw, dict):
        return []
    atoms = []
    for item in raw.get("macro_atoms", []):
        inputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("inputs", [])
        ]
        outputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("outputs", [])
        ]
        try:
            concept = ConceptType(item.get("concept_type", "custom"))
        except ValueError:
            logger.warning(
                "Unknown concept_type %r for node %r, falling back to custom",
                item.get("concept_type"),
                item.get("name"),
            )
            concept = ConceptType.CUSTOM
        atoms.append(
            MacroAtomSpec(
                name=item.get("name", ""),
                description=item.get("description", ""),
                method_names=item.get("method_names", []),
                inputs=inputs,
                outputs=outputs,
                config_params=item.get("config_params", []),
                concept_type=concept,
                is_optional=item.get("is_optional", False),
                is_stochastic=item.get("is_stochastic", False),
                requires_rng_key=item.get("requires_rng_key", False),
                requires_autodiff=item.get("requires_autodiff", False),
                autodiff_backend=item.get("autodiff_backend", ""),
            )
        )
    return atoms


def _parse_edges(raw: dict) -> list[DependencyEdge]:
    if not isinstance(raw, dict):
        return []
    edges = []
    for item in raw.get("edges", []):
        edges.append(
            DependencyEdge(
                source_id=item.get("source_id", ""),
                target_id=item.get("target_id", ""),
                output_name=item.get("output_name", ""),
                input_name=item.get("input_name", ""),
                source_type=item.get("source_type", ""),
                target_type=item.get("target_type", ""),
            )
        )
    return edges


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------


async def propose_macro_atoms(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """LLM call: group methods into macro-atoms."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="propose_macro_atoms")
    dfg = state["raw_dfg"]

    # Opaque DL boundary: deterministic single-atom plan, no LLM cost
    if dfg.is_opaque and dfg.methods:
        mf = dfg.methods[0]
        atom = MacroAtomSpec(
            name=dfg.class_name,
            description=mf.docstring or f"Opaque DL boundary: {dfg.class_name}",
            method_names=[mf.name],
            inputs=[IOSpec(name=p, type_desc="Any") for p in mf.params],
            outputs=[IOSpec(name="output", type_desc=mf.return_type or "Any")],
            concept_type=ConceptType.NEURAL_NETWORK,
            is_opaque=True,
        )
        plan = ProposedMacroPlan(macro_atoms=[atom])
        return {"proposed_plan": _attach_canonical_ir(dfg, plan)}

    if _is_simple_class(dfg):
        return {"proposed_plan": _attach_canonical_ir(dfg, _chunk_by_method(dfg))}

    retry_context = ""
    if state.get("retry_count", 0) > 0:
        missing = state.get("missing_attrs", [])
        retry_context = (
            f"RETRY {state['retry_count']}/{_MAX_RETRIES}. "
            f"Previous attempt missed these attributes: {missing}. "
            f"Every self.* attribute MUST appear in at least one macro-atom."
        )

    user_prompt = SEMANTIC_CHUNK_USER.format(
        class_name=dfg.class_name,
        method_summaries=_build_method_summaries(dfg),
        attr_graph=_build_attr_graph(dfg),
        config_branches=_build_config_branches(dfg),
        retry_context=retry_context,
    )

    if mon:
        mon.llm_start(INGESTER_CHUNK)
    try:
        response = await select_llm(deps.llm, INGESTER_CHUNK).complete(
            SEMANTIC_CHUNK_SYSTEM, user_prompt
        )
        if mon:
            mon.llm_end(INGESTER_CHUNK, ok=True)
    except Exception as exc:
        if mon:
            mon.llm_end(INGESTER_CHUNK, ok=False, error=str(exc))
        raise

    try:
        raw = extract_json(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse LLM response as JSON, using empty plan")
        raw = {"macro_atoms": [], "edges": []}

    macro_atoms = _parse_macro_atoms(raw)
    edges = _parse_edges(raw)

    plan = ProposedMacroPlan(
        macro_atoms=macro_atoms,
        edge_definitions=edges,
    )
    return {"proposed_plan": _attach_canonical_ir(dfg, plan)}


async def flatten_config(state: ChunkerState, config: RunnableConfig) -> dict[str, Any]:
    """Deterministic: Flatten config-gated branches into optional variants."""
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="flatten_config")
    plan = state["proposed_plan"]
    dfg = state["raw_dfg"]

    new_atoms = []
    for atom in plan.macro_atoms:
        branches = []
        for mname in atom.method_names:
            mf = next((m for m in dfg.methods if m.name == mname), None)
            if mf:
                branches.extend(mf.config_branches)

        if not branches:
            new_atoms.append(atom)
            continue

        # Analyze branches to see if we should split
        # For now, we simply ensure the atom is marked is_optional if it has significant branching
        # and append it. In a full implementation, we would duplicate the atom for each variant.
        # Here we simulate the flattening by marking it for the orchestrator.
        if len(branches) > 0 and not atom.is_optional:
            # It has config branches but wasn"t marked optional.
            # We keep it as is but could tag it in description.
            atom.description += f" (Contains {len(branches)} config branches)"
            # atom.is_optional = True # Optional implies it might not run. Branching implies one of many runs.

        new_atoms.append(atom)

    updated = plan.model_copy(update={"macro_atoms": new_atoms})
    updated = _attach_canonical_ir(dfg, updated)
    return {"proposed_plan": updated}


async def hoist_state(state: ChunkerState, config: RunnableConfig) -> dict[str, Any]:
    """LLM call: identify cross-window attrs and generate state model specs."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="hoist_state")
    dfg = state["raw_dfg"]
    plan = state["proposed_plan"]

    if not dfg.cross_window_attrs:
        return {"proposed_plan": _attach_canonical_ir(dfg, plan)}

    macro_plan_json = json.dumps([a.model_dump() for a in plan.macro_atoms], indent=2)
    user_prompt = HOIST_STATE_USER.format(
        cross_window_attrs=dfg.cross_window_attrs,
        macro_plan_json=macro_plan_json,
    )

    if mon:
        mon.llm_start(INGESTER_HOIST_STATE)
    try:
        response = await select_llm(deps.llm, INGESTER_HOIST_STATE).complete(
            HOIST_STATE_SYSTEM, user_prompt
        )
        if mon:
            mon.llm_end(INGESTER_HOIST_STATE, ok=True)
    except Exception as exc:
        if mon:
            mon.llm_end(INGESTER_HOIST_STATE, ok=False, error=str(exc))
        raise

    try:
        raw = extract_json(response)
    except json.JSONDecodeError:
        logger.warning("Failed to parse state hoisting response")
        return {"proposed_plan": _attach_canonical_ir(dfg, plan)}

    state_models = []
    for item in raw.get("state_models", []):
        fields = [tuple(f) for f in item.get("fields", [])]
        state_models.append(
            StateModelSpec(
                model_name=item.get("model_name", ""),
                fields=fields,
                source_attrs=item.get("source_attrs", []),
                docstring=item.get("docstring", ""),
            )
        )

    updated = plan.model_copy(update={"state_models": state_models})

    # Compute deterministic state edges and append to existing edges
    state_edges = _compute_state_edges(dfg, updated)
    if state_edges:
        all_edges = list(updated.edge_definitions) + state_edges
        updated = updated.model_copy(update={"edge_definitions": all_edges})

    return {"proposed_plan": _attach_canonical_ir(dfg, updated)}


async def search_sub_atoms(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Deterministic: query FAISS index for existing atoms matching operations."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="search_sub_atoms")
    plan = state["proposed_plan"]

    if deps.faiss_index is None:
        return {"proposed_plan": plan}

    sub_refs: list[SubAtomRef] = []
    for atom in plan.macro_atoms:
        results = deps.faiss_index.search_by_embedding(atom.name, k=3)
        for decl, score in results:
            if score > 0.5:
                sub_refs.append(
                    SubAtomRef(
                        atom_name=decl.name,
                        similarity_score=score,
                    )
                )

    updated = plan.model_copy(update={"sub_atom_refs": sub_refs})
    return {"proposed_plan": updated}


async def critic_validate(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Validate that ALL self.* attributes appear in macro-atoms or state models."""
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="critic_validate")
    dfg = state["raw_dfg"]
    plan = state["proposed_plan"]

    # Opaque DL boundary: auto-pass (no self.* tracking)
    if dfg.is_opaque:
        return {
            "validated_plan": ValidatedMacroPlan(
                plan=plan,
                all_attrs_accounted=True,
                coverage_report="Opaque DL boundary: no self.* tracking required.",
            ),
            "critique_passed": True,
            "critique_reason": "",
        }

    canonical_plan = plan if plan.canonical_ir is not None else _attach_canonical_ir(dfg, plan)
    ir = canonical_plan.canonical_ir or _build_ingest_ir(dfg, canonical_plan)
    ok, ir_report, missing = _validate_canonical_ir(dfg, ir)

    if ok:
        validated = ValidatedMacroPlan(
            plan=canonical_plan,
            all_attrs_accounted=True,
            coverage_report="All attributes accounted for.",
            ir_validated=True,
            ir_coverage_report=ir_report,
        )
        return {
            "validated_plan": validated,
            "critique_passed": True,
            "critique_reason": "",
        }
    else:
        missing_list = sorted(missing)
        return {
            "critique_passed": False,
            "critique_reason": ir_report,
            "missing_attrs": missing_list,
        }


async def prepare_chunk_retry(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Increment retry counter for the next proposal attempt."""
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="prepare_chunk_retry")
    return {"retry_count": state.get("retry_count", 0) + 1}


# ---------------------------------------------------------------------------
# Complexity heuristic + recursive decomposition
# ---------------------------------------------------------------------------


def is_atom_complex(
    atom: MacroAtomSpec,
    dfg: RawDataFlowGraph,
    line_threshold: int = 30,
) -> bool:
    """Determine whether *atom* is too complex and should be decomposed.

    An atom is complex when ANY of:
    - Combined method source exceeds *line_threshold* lines
    - Methods call 3+ internal sub-functions
    - Any method body is ``NotImplementedError`` (skeleton stub)
    """
    total_lines = 0
    total_internal_calls = 0
    has_not_implemented = False

    by_name = {m.name: m for m in dfg.methods}
    for mname in atom.method_names:
        mf = by_name.get(mname)
        if mf is None:
            continue
        src = mf.source_code or ""
        total_lines += len(src.strip().splitlines())
        # Count calls to other methods in the same class
        internal_methods = {m.name for m in dfg.methods}
        total_internal_calls += sum(1 for c in mf.calls if c in internal_methods)
        if "NotImplementedError" in src:
            has_not_implemented = True

    # Also honour the pre-populated source_lines field
    if atom.source_lines > 0:
        total_lines = max(total_lines, atom.source_lines)

    return total_lines > line_threshold or total_internal_calls >= 3 or has_not_implemented


def _gather_source_for_atom(
    atom: MacroAtomSpec,
    dfg: RawDataFlowGraph,
) -> tuple[str, list[str]]:
    """Return (combined_source, internal_calls) for an atom's methods."""
    by_name = {m.name: m for m in dfg.methods}
    sources: list[str] = []
    all_calls: list[str] = []
    for mname in atom.method_names:
        mf = by_name.get(mname)
        if mf and mf.source_code:
            sources.append(mf.source_code)
            all_calls.extend(mf.calls)
    return "\n\n".join(sources), all_calls


def _parse_sub_atoms(raw: dict, parent_depth: int) -> tuple[list[MacroAtomSpec], list[DependencyEdge]]:
    """Parse LLM decomposition response into sub-atoms and edges."""
    sub_atoms: list[MacroAtomSpec] = []
    for item in raw.get("sub_atoms", []):
        inputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("inputs", [])
        ]
        outputs = [
            IOSpec(
                name=io.get("name", ""),
                type_desc=io.get("type_desc", ""),
                constraints=io.get("constraints", ""),
            )
            for io in item.get("outputs", [])
        ]
        try:
            concept = ConceptType(item.get("concept_type", "custom"))
        except ValueError:
            logger.warning(
                "Unknown concept_type %r for node %r, falling back to custom",
                item.get("concept_type"),
                item.get("name"),
            )
            concept = ConceptType.CUSTOM
        sub_atoms.append(
            MacroAtomSpec(
                name=item.get("name", ""),
                description=item.get("description", ""),
                inputs=inputs,
                outputs=outputs,
                concept_type=concept,
                depth=parent_depth + 1,
            )
        )

    edges: list[DependencyEdge] = []
    for item in raw.get("edges", []):
        edges.append(
            DependencyEdge(
                source_id=_snake_case_id(item.get("source_id", "")),
                target_id=_snake_case_id(item.get("target_id", "")),
                output_name=item.get("output_name", ""),
                input_name=item.get("input_name", ""),
                source_type=item.get("source_type", ""),
                target_type=item.get("target_type", ""),
            )
        )
    return sub_atoms, edges


def _snake_case_id(name: str) -> str:
    """Normalize an edge endpoint to snake_case (matching _emit_atom_nodes)."""
    return name.lower().replace(" ", "_").replace("-", "_")


def _merge_iospec_lists(primary: list[IOSpec], secondary: list[IOSpec]) -> list[IOSpec]:
    """Merge IOSpec lists by port name, keeping richer type/constraint details."""
    merged: dict[str, IOSpec] = {}
    order: list[str] = []
    for spec in [*primary, *secondary]:
        key = spec.name
        if key not in merged:
            merged[key] = spec
            order.append(key)
            continue
        existing = merged[key]
        merged[key] = IOSpec(
            name=existing.name or spec.name,
            type_desc=existing.type_desc or spec.type_desc,
            constraints=existing.constraints or spec.constraints,
        )
    return [merged[key] for key in order]


def _dedupe_dependency_edges(edges: list[DependencyEdge]) -> list[DependencyEdge]:
    """Deduplicate dependency edges by full structural key."""
    seen: set[tuple[str, str, str, str, str, str]] = set()
    deduped: list[DependencyEdge] = []
    for edge in edges:
        key = (
            edge.source_id,
            edge.target_id,
            edge.output_name,
            edge.input_name,
            edge.source_type,
            edge.target_type,
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(edge)
    return deduped


def _merge_macro_atoms(existing: MacroAtomSpec, incoming: MacroAtomSpec) -> MacroAtomSpec:
    """Merge duplicate macro-atoms while preserving the richer combined spec."""
    concept_type = existing.concept_type
    if concept_type == ConceptType.CUSTOM and incoming.concept_type != ConceptType.CUSTOM:
        concept_type = incoming.concept_type

    description = existing.description
    if len(incoming.description.strip()) > len(description.strip()):
        description = incoming.description

    conceptual_profile = existing.conceptual_profile or incoming.conceptual_profile
    if (
        existing.conceptual_profile is not None
        and incoming.conceptual_profile is not None
        and len(incoming.conceptual_profile.model_dump_json())
        > len(existing.conceptual_profile.model_dump_json())
    ):
        conceptual_profile = incoming.conceptual_profile

    return existing.model_copy(
        update={
            "description": description,
            "method_names": list(
                dict.fromkeys([*existing.method_names, *incoming.method_names])
            ),
            "inputs": _merge_iospec_lists(existing.inputs, incoming.inputs),
            "outputs": _merge_iospec_lists(existing.outputs, incoming.outputs),
            "config_params": list(
                dict.fromkeys([*existing.config_params, *incoming.config_params])
            ),
            "decorators": list(
                dict.fromkeys([*existing.decorators, *incoming.decorators])
            ),
            "concept_type": concept_type,
            "is_optional": existing.is_optional or incoming.is_optional,
            "is_opaque": existing.is_opaque or incoming.is_opaque,
            "is_external": existing.is_external or incoming.is_external,
            "is_stochastic": existing.is_stochastic or incoming.is_stochastic,
            "requires_rng_key": existing.requires_rng_key or incoming.requires_rng_key,
            "requires_autodiff": existing.requires_autodiff or incoming.requires_autodiff,
            "autodiff_backend": existing.autodiff_backend or incoming.autodiff_backend,
            "conceptual_profile": conceptual_profile,
            "children": _dedupe_macro_atoms([*existing.children, *incoming.children]),
            "sub_edges": _dedupe_dependency_edges(
                [*existing.sub_edges, *incoming.sub_edges]
            ),
            "depth": min(existing.depth, incoming.depth),
            "source_lines": max(existing.source_lines, incoming.source_lines),
        }
    )


def _dedupe_macro_atoms(atoms: list[MacroAtomSpec]) -> list[MacroAtomSpec]:
    """Deduplicate macro-atoms by canonical node id, recursively merging children."""
    merged: dict[str, MacroAtomSpec] = {}
    order: list[str] = []
    for atom in atoms:
        normalized = atom.model_copy(
            update={
                "children": _dedupe_macro_atoms(atom.children),
                "sub_edges": _dedupe_dependency_edges(atom.sub_edges),
            }
        )
        key = _snake_case_id(normalized.name)
        if key not in merged:
            merged[key] = normalized
            order.append(key)
            continue
        merged[key] = _merge_macro_atoms(merged[key], normalized)
    return [merged[key] for key in order]


async def _decompose_single_atom(
    atom: MacroAtomSpec,
    dfg: RawDataFlowGraph,
    llm: LLMClient,
    max_depth: int,
    line_threshold: int,
    current_depth: int,
    monitor: IngestMonitor | None = None,
    shared_context: SharedContextStore | None = None,
    shared_context_metrics: SharedContextMetrics | None = None,
    context_namespace: str = "",
    context_budget_chars: int = 900,
    ancestor_ids: tuple[str, ...] = (),
) -> MacroAtomSpec:
    """Recursively decompose a single atom if it exceeds complexity thresholds."""
    if current_depth >= max_depth:
        return atom

    if not is_atom_complex(atom, dfg, line_threshold):
        return atom

    source_code, internal_calls = _gather_source_for_atom(atom, dfg)
    if not source_code.strip():
        return atom

    # Try deterministic control-flow decomposition first
    from sciona.ingester.control_flow_decomposer import decompose_function

    primary_method = atom.method_names[0] if atom.method_names else atom.name
    cf_result = decompose_function(source_code, primary_method)
    if cf_result is not None and cf_result.confidence >= 0.5:
        children = []
        for sub in cf_result.sub_atoms:
            child = MacroAtomSpec(
                name=sub.name,
                description=sub.description,
                method_names=[],
                inputs=[IOSpec(name=i, type_desc="") for i in sub.inputs],
                outputs=[IOSpec(name=o, type_desc="") for o in sub.outputs],
                concept_type=atom.concept_type,
                depth=current_depth + 1,
                source_lines=list(range(sub.source_lines[0], sub.source_lines[1] + 1)),
            )
            children.append(child)
        if children:
            atom.children = children
            atom.sub_edges = [
                (e["from"], e["to"], e.get("data", ""))
                for e in cf_result.edges
            ]
            logger.info(
                "Deterministic CFG decomposition for %s: %d sub-atoms (confidence=%.2f)",
                atom.name,
                len(children),
                cf_result.confidence,
            )
            return atom

    inputs_str = ", ".join(f"{i.name}: {i.type_desc}" for i in atom.inputs) or "(none)"
    outputs_str = ", ".join(f"{o.name}: {o.type_desc}" for o in atom.outputs) or "(none)"
    calls_str = ", ".join(internal_calls) if internal_calls else "(none)"

    user_prompt = DECOMPOSE_ATOM_USER.format(
        atom_name=atom.name,
        atom_description=atom.description,
        current_inputs=inputs_str,
        current_outputs=outputs_str,
        internal_calls=calls_str,
        source_code=source_code,
    )
    if shared_context is not None and context_namespace:
        try:
            records = await shared_context.search(
                f"{context_namespace}/decompose",
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
        except Exception:
            pass

    try:
        if monitor:
            monitor.llm_start(INGESTER_DECOMPOSE)
        response = await select_llm(llm, INGESTER_DECOMPOSE).complete(
            DECOMPOSE_ATOM_SYSTEM, user_prompt
        )
    except Exception as exc:
        if monitor:
            monitor.llm_end(INGESTER_DECOMPOSE, ok=False, error=str(exc))
        logger.warning("Decomposition failed for %s: %s", atom.name, exc)
        return atom

    if monitor:
        monitor.llm_end(INGESTER_DECOMPOSE, ok=True)
    try:
        raw = extract_json(response)
    except Exception as exc:
        logger.warning("Decomposition JSON parse failed for %s: %s", atom.name, exc)
        return atom

    children, sub_edges = _parse_sub_atoms(raw, current_depth)
    children = _dedupe_macro_atoms(children)
    child_ids = {_snake_case_id(child.name) for child in children}
    sub_edges = [
        edge
        for edge in _dedupe_dependency_edges(sub_edges)
        if edge.source_id in child_ids and edge.target_id in child_ids
    ]
    if not children:
        return atom
    if shared_context is not None and context_namespace:
        try:
            await shared_context.put(
                f"{context_namespace}/decompose",
                (
                    f"Atom: {atom.name}\n"
                    f"Children: {', '.join(child.name for child in children)}\n"
                    f"Sub-edge count: {len(sub_edges)}"
                ),
                metadata={"atom_name": atom.name, "depth": current_depth},
            )
        except Exception:
            pass

    # Recurse into children
    recursed_children: list[MacroAtomSpec] = []
    next_ancestor_ids = (*ancestor_ids, _snake_case_id(atom.name))
    for child in children:
        child_id = _snake_case_id(child.name)
        if child_id in next_ancestor_ids:
            logger.info(
                "Skipping duplicate/cyclic sub-atom %s under %s",
                child.name,
                atom.name,
            )
            continue
        recursed = await _decompose_single_atom(
            child,
            dfg,
            llm,
            max_depth,
            line_threshold,
            current_depth + 1,
            monitor=monitor,
            shared_context=shared_context,
            shared_context_metrics=shared_context_metrics,
            context_namespace=context_namespace,
            context_budget_chars=context_budget_chars,
            ancestor_ids=next_ancestor_ids,
        )
        recursed_children.append(recursed)

    return atom.model_copy(
        update={
            "children": _dedupe_macro_atoms(recursed_children),
            "sub_edges": sub_edges,
        }
    )


async def decompose_complex_atoms(
    state: ChunkerState, config: RunnableConfig
) -> dict[str, Any]:
    """Recursively decompose complex atoms based on complexity heuristics."""
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="decompose_complex_atoms")
    validated = state["validated_plan"]
    plan = validated.plan
    dfg = state["raw_dfg"]

    max_depth = deps.max_depth
    line_threshold = deps.line_threshold

    # At depth 1 (default), skip decomposition entirely for backward compat
    if max_depth <= 1:
        return {"validated_plan": validated}

    parallelism = max(1, deps.parallelism)
    if parallelism <= 1 or len(plan.macro_atoms) <= 1:
        decomposed_atoms: list[MacroAtomSpec] = []
        for atom in plan.macro_atoms:
            result = await _decompose_single_atom(
                atom,
                dfg,
                deps.llm,
                max_depth,
                line_threshold,
                current_depth=1,
                monitor=mon,
                shared_context=deps.shared_context,
                shared_context_metrics=deps.shared_context_metrics,
                context_namespace=deps.context_namespace,
                context_budget_chars=deps.context_budget_chars,
            )
            decomposed_atoms.append(result)
    else:
        semaphore = asyncio.Semaphore(parallelism)

        async def _run(atom: MacroAtomSpec) -> MacroAtomSpec:
            async with semaphore:
                return await _decompose_single_atom(
                    atom,
                    dfg,
                    deps.llm,
                    max_depth,
                    line_threshold,
                    current_depth=1,
                    monitor=mon,
                    shared_context=deps.shared_context,
                    shared_context_metrics=deps.shared_context_metrics,
                    context_namespace=deps.context_namespace,
                    context_budget_chars=deps.context_budget_chars,
                )

        decomposed_atoms = list(await asyncio.gather(*[_run(a) for a in plan.macro_atoms]))

    new_plan = plan.model_copy(update={"macro_atoms": _dedupe_macro_atoms(decomposed_atoms)})
    new_plan = _attach_canonical_ir(dfg, new_plan)
    new_validated = validated.model_copy(update={"plan": new_plan})
    return {"validated_plan": new_validated}


def _format_io_specs(specs: list) -> str:
    """Format IOSpec list for the abstraction prompt."""
    if not specs:
        return "(none)"
    return "\n".join(
        f"  - {s.name}: {s.type_desc}"
        + (f" ({s.constraints})" if s.constraints else "")
        for s in specs
    )


def _build_enriched_description(original: str, profile: ConceptualProfile) -> str:
    """Merge a ConceptualProfile into an atom description.

    The profile JSON is appended as a fenced block so downstream consumers
    (the Hunter's FAISS index, CDG node descriptions, generated docstrings)
    all benefit from the domain-agnostic vocabulary.
    """
    profile_json = json.dumps(profile.model_dump(), indent=2)
    return f"{original}\n\n<!-- conceptual_profile -->\n{profile_json}\n<!-- /conceptual_profile -->"


def _parse_conceptual_profile(raw: dict) -> ConceptualProfile:
    """Parse a raw LLM JSON response into a ConceptualProfile."""
    return ConceptualProfile(
        abstract_name=raw.get("abstract_name", ""),
        conceptual_transform=raw.get("conceptual_transform", ""),
        abstract_inputs=raw.get("abstract_inputs", []),
        abstract_outputs=raw.get("abstract_outputs", []),
        algorithmic_properties=raw.get("algorithmic_properties", []),
        cross_disciplinary_applications=raw.get("cross_disciplinary_applications", []),
    )


async def abstract_atoms(state: ChunkerState, config: RunnableConfig) -> dict[str, Any]:
    """LLM call: generate domain-agnostic conceptual profiles for each atom.

    Runs the Conceptual Abstraction Agent on every atom in the validated
    plan.  The resulting profile is stored on each atom's
    ``conceptual_profile`` field.  A plain-text summary is threaded
    through to the FAISS index via the emitter (not embedded in the
    description).
    """
    deps: ChunkerDeps = config["configurable"]["deps"]
    mon = _get_monitor(config)
    if mon:
        mon.heartbeat(phase="phase2_chunk", step="abstract_atoms")
    validated = state["validated_plan"]
    plan = validated.plan

    async def _abstract_one(atom: MacroAtomSpec) -> MacroAtomSpec:
        user_prompt = CONCEPTUAL_ABSTRACT_USER.format(
            atom_name=atom.name,
            atom_description=atom.description,
            concept_type=atom.concept_type.value,
            inputs_spec=_format_io_specs(atom.inputs),
            outputs_spec=_format_io_specs(atom.outputs),
            method_names=", ".join(atom.method_names),
        )
        shared_block = await _search_context(
            deps,
            channel="abstract",
            query=f"{atom.name} {atom.concept_type.value}",
            limit=3,
        )
        if shared_block:
            user_prompt += f"\n\n{shared_block}"

        try:
            if mon:
                mon.llm_start(INGESTER_ABSTRACT)
            response = await select_llm(deps.llm, INGESTER_ABSTRACT).complete(
                CONCEPTUAL_ABSTRACT_SYSTEM, user_prompt
            )
            if mon:
                mon.llm_end(INGESTER_ABSTRACT, ok=True)
        except Exception as exc:
            if mon:
                mon.llm_end(INGESTER_ABSTRACT, ok=False, error=str(exc))
            logger.warning("Conceptual abstraction failed for %s: %s", atom.name, exc)
            profile = ConceptualProfile(abstract_name=atom.name)
            return atom.model_copy(update={"conceptual_profile": profile})

        try:
            raw = extract_json(response)
            profile = _parse_conceptual_profile(raw)
        except Exception as exc:
            logger.warning(
                "Conceptual abstraction JSON parse failed for %s: %s", atom.name, exc
            )
            profile = ConceptualProfile(abstract_name=atom.name)

        await _put_context(
            deps,
            channel="abstract",
            text=(
                f"Atom: {atom.name}\n"
                f"Abstract name: {profile.abstract_name}\n"
                f"Transform: {profile.conceptual_transform}"
            ),
            metadata={"atom_name": atom.name, "concept_type": atom.concept_type.value},
        )
        return atom.model_copy(
            update={
                "conceptual_profile": profile,
            }
        )

    parallelism = max(1, deps.parallelism)
    if parallelism <= 1 or len(plan.macro_atoms) <= 1:
        enriched_atoms = [await _abstract_one(atom) for atom in plan.macro_atoms]
    else:
        semaphore = asyncio.Semaphore(parallelism)

        async def _run(atom: MacroAtomSpec) -> MacroAtomSpec:
            async with semaphore:
                return await _abstract_one(atom)

        enriched_atoms = list(await asyncio.gather(*[_run(a) for a in plan.macro_atoms]))

    new_plan = plan.model_copy(update={"macro_atoms": enriched_atoms})
    new_plan = _attach_canonical_ir(state["raw_dfg"], new_plan)
    new_validated = validated.model_copy(update={"plan": new_plan})
    return {"validated_plan": new_validated}


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------


def route_after_critic(state: ChunkerState) -> str:
    if state.get("critique_passed", False):
        return "end"
    if state.get("retry_count", 0) >= _MAX_RETRIES:
        return "end_best_effort"
    return "retry"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------


def build_chunker_graph() -> StateGraph:
    """Construct the Phase 2 semantic chunking sub-graph.

    Flow::

        propose_macro_atoms -> flatten_config -> hoist_state
        -> search_sub_atoms -> critic_validate
        -> [decompose_complex_atoms -> abstract_atoms -> END
            | prepare_chunk_retry -> propose_macro_atoms]

    The ``decompose_complex_atoms`` step recursively splits complex atoms
    into sub-atoms (controlled by ``max_depth`` / ``line_threshold`` in deps).
    The ``abstract_atoms`` step runs the Conceptual Abstraction Agent on
    each atom after the plan is finalized (critic passed or budget exhausted),
    storing domain-agnostic profiles for cross-field semantic retrieval.
    """
    graph = StateGraph(ChunkerState)

    graph.add_node("propose_macro_atoms", propose_macro_atoms)
    graph.add_node("flatten_config", flatten_config)
    graph.add_node("hoist_state", hoist_state)
    graph.add_node("search_sub_atoms", search_sub_atoms)
    graph.add_node("critic_validate", critic_validate)
    graph.add_node("prepare_chunk_retry", prepare_chunk_retry)
    graph.add_node("decompose_complex_atoms", decompose_complex_atoms)
    graph.add_node("abstract_atoms", abstract_atoms)

    graph.set_entry_point("propose_macro_atoms")
    graph.add_edge("propose_macro_atoms", "flatten_config")
    graph.add_edge("flatten_config", "hoist_state")
    graph.add_edge("hoist_state", "search_sub_atoms")
    graph.add_edge("search_sub_atoms", "critic_validate")

    graph.add_conditional_edges(
        "critic_validate",
        route_after_critic,
        {
            "end": "decompose_complex_atoms",
            "end_best_effort": "decompose_complex_atoms",
            "retry": "prepare_chunk_retry",
        },
    )
    graph.add_edge("decompose_complex_atoms", "abstract_atoms")
    graph.add_edge("abstract_atoms", END)
    graph.add_edge("prepare_chunk_retry", "propose_macro_atoms")

    return graph

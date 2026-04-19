"""CDG (Conceptual Dependency Graph) IR validation.

Validates that a canonical IR plan is consistent with the data-flow graph
it was derived from. Checks operation coverage, state slot validity,
fitted/config consistency, attribute coverage, and cycle detection.

This module contains a copy of the validation logic from the ingester
chunker to avoid pulling in the chunker's heavy module-level imports
(langgraph, LLM clients, etc.).
"""

from __future__ import annotations

from sciona.architect.models import ConceptType
from sciona.ingester.models import IngestIRPlan, RawDataFlowGraph
from sciona.tools.cycle import detect_cycles


def validate_cdg_ir(
    dfg: RawDataFlowGraph,
    ir: IngestIRPlan,
) -> tuple[bool, str, list[str]]:
    """Validate that a canonical IR plan is consistent with its source DFG.

    Checks:
    - All operations have method bindings
    - Required state slots are defined
    - Query/metadata operations do not mutate state
    - Output bindings have source methods and non-unknown kinds
    - Fitted attributes are not downgraded to transient
    - Config slots have constructor provenance
    - All source attributes are covered
    - No non-message-passing cycles exist

    Args:
        dfg: The raw data-flow graph from AST extraction.
        ir: The canonical IR plan to validate.

    Returns:
        (ok, report, missing_attrs) tuple where ok is True if all
        checks pass, report is a human-readable summary, and
        missing_attrs lists uncovered attribute names.
    """
    issues: list[str] = []
    slot_names = {slot.slot_name for slot in ir.state_slots}
    covered_methods = {
        binding.method_name
        for operation in ir.operations
        for binding in operation.method_bindings
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
                issues.append(
                    f"{operation.display_name}: unknown required state slot {required_slot}"
                )
        if operation.role in {"query", "metadata", "score"}:
            mutating = [
                effect.slot_name
                for effect in operation.state_effects
                if effect.effect_kind in {"initialize", "update", "clear"}
            ]
            if mutating:
                issues.append(
                    f"{operation.display_name}: {operation.role} operation mutates "
                    f"state {sorted(mutating)}"
                )
        for binding in operation.emitted_outputs:
            if not binding.source_method:
                issues.append(
                    f"{operation.display_name}: output {binding.output_name} "
                    "missing source method"
                )
            if binding.binding_kind == "unknown":
                issues.append(
                    f"{operation.display_name}: output {binding.output_name} "
                    "has unknown binding"
                )
            if binding.source_attr:
                covered_attrs.add(binding.source_attr)

    for slot in ir.state_slots:
        if slot.state_kind == "transient" and slot.slot_name in set(
            dfg.fitted_attributes
        ):
            issues.append(f"{slot.slot_name}: fitted attribute downgraded to transient")
        if slot.state_kind == "config" and slot.slot_name not in set(
            dfg.config_attributes
        ):
            attr_roles = {
                method_to_role.get(method_name, "")
                for method_name in [*slot.read_by, *slot.written_by]
            }
            if "constructor" not in attr_roles:
                issues.append(
                    f"{slot.slot_name}: config slot lacks constructor provenance"
                )

    missing_attrs = sorted(set(dfg.all_attributes) - covered_attrs)
    if missing_attrs:
        issues.append(f"Missing attributes: {missing_attrs}")

    operation_ids = {operation.operation_id for operation in ir.operations}
    cycle_nodes = detect_cycles(
        operation_ids,
        [
            (edge.source_operation_id, edge.target_operation_id)
            for edge in ir.edges
        ],
    )
    if cycle_nodes:
        op_by_id = {op.operation_id: op for op in ir.operations}
        if not all(
            op_by_id[op_id].concept_type == ConceptType.MESSAGE_PASSING
            for op_id in cycle_nodes
            if op_id in op_by_id
        ):
            issues.append(
                f"Non-message-passing operation cycle: {sorted(cycle_nodes)}"
            )

    report = "All canonical IR checks passed." if not issues else "; ".join(issues)
    return not issues, report, missing_attrs


__all__ = ["validate_cdg_ir"]

"""Pydantic models for the Smart Ingester (Round 0).

Converts existing stateful Python classes into stateless atom graphs
compatible with the AGEO framework.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from sciona.architect.handoff import CDGExport
from sciona.architect.models import ConceptType, DependencyEdge, IOSpec
from sciona.types import MatchResult

# ---------------------------------------------------------------------------
# Phase 1: AST extraction outputs
# ---------------------------------------------------------------------------


class AttributeAccess(BaseModel):
    """A single ``self.X`` read or write traced from AST analysis."""

    attr_name: str
    access_type: str  # "read" | "write"
    method_name: str
    line_number: int = 0
    is_config: bool = False


class ConfigBranch(BaseModel):
    """A config-gated branch: ``if self.options.X``."""

    config_attr: str
    method: str
    lines: tuple[int, int] = (0, 0)
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)


class SourceSpan(BaseModel):
    """Concrete source location for an extracted fact."""

    file_path: str = ""
    line_start: int = 0
    line_end: int = 0
    col_start: int = 0
    col_end: int = 0


class FactProvenance(BaseModel):
    """Deterministic evidence supporting an extracted fact."""

    rule_id: str = ""
    span: SourceSpan = Field(default_factory=SourceSpan)
    evidence: str = ""


class UnknownFact(BaseModel):
    """A place where extraction found ambiguity but refused to guess."""

    reason: str
    detail: str = ""
    provenance: FactProvenance = Field(default_factory=FactProvenance)


class ParameterFact(BaseModel):
    """Exact parameter information for a callable signature."""

    name: str
    kind: str = "positional_or_keyword"
    annotation: str = ""
    default_expression: str = ""
    has_default: bool = False
    provenance: FactProvenance = Field(default_factory=FactProvenance)


class ReturnFact(BaseModel):
    """A normalized summary of one observed return path."""

    kind: str = "unknown"
    expression: str = ""
    referenced_attrs: list[str] = Field(default_factory=list)
    referenced_callees: list[str] = Field(default_factory=list)
    provenance: FactProvenance = Field(default_factory=FactProvenance)


class CallFact(BaseModel):
    """One observed call site inside a method body."""

    callee_expression: str = ""
    resolved_target: str = ""
    args: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    provenance: FactProvenance = Field(default_factory=FactProvenance)


class AttributeSemanticFact(BaseModel):
    """Aggregated semantic inventory for one attribute."""

    attr_name: str
    first_seen_in: str = ""
    read_methods: list[str] = Field(default_factory=list)
    write_methods: list[str] = Field(default_factory=list)
    is_config: bool = False
    is_fitted: bool = False
    is_derived: bool = False
    is_query_only: bool = False
    provenances: list[FactProvenance] = Field(default_factory=list)


class MethodBinding(BaseModel):
    """Source-faithful binding between an IR operation and an upstream method."""

    method_name: str
    signature: list[ParameterFact] = Field(default_factory=list)
    call_style: str = ""
    return_behavior: list[ReturnFact] = Field(default_factory=list)
    requires_instance_state: bool = False
    provenance: list[FactProvenance] = Field(default_factory=list)


class OutputBindingSpec(BaseModel):
    """A concrete output emitted by an IR operation."""

    output_name: str
    type_desc: str = "Any"
    binding_kind: str = "unknown"
    source_method: str = ""
    source_attr: str = ""
    tuple_index: int | None = None
    provenance: list[FactProvenance] = Field(default_factory=list)


class StateSlotSpec(BaseModel):
    """Canonical IR representation of one state slot."""

    slot_name: str
    state_kind: str = "transient"
    type_desc: str = "Any"
    required_before: list[str] = Field(default_factory=list)
    written_by: list[str] = Field(default_factory=list)
    read_by: list[str] = Field(default_factory=list)
    source_attr: str = ""
    provenance: list[FactProvenance] = Field(default_factory=list)


class StateEffectSpec(BaseModel):
    """State mutation or read-only effect performed by an operation."""

    slot_name: str
    effect_kind: str = "unknown"
    source_method: str = ""
    provenance: list[FactProvenance] = Field(default_factory=list)


class OperationEdge(BaseModel):
    """Typed edge between canonical IR operations."""

    source_operation_id: str
    target_operation_id: str
    edge_kind: str = "data"
    artifact_or_slot_name: str = ""
    provenance: list[FactProvenance] = Field(default_factory=list)


class OperationSpec(BaseModel):
    """Canonical OO-aware ingest operation."""

    operation_id: str
    display_name: str
    role: str = "unknown"
    method_bindings: list[MethodBinding] = Field(default_factory=list)
    direct_inputs: list[IOSpec] = Field(default_factory=list)
    required_state_slots: list[str] = Field(default_factory=list)
    emitted_outputs: list[OutputBindingSpec] = Field(default_factory=list)
    state_effects: list[StateEffectSpec] = Field(default_factory=list)
    concept_type: ConceptType = ConceptType.CUSTOM
    is_optional: bool = False
    is_opaque: bool = False
    is_external: bool = False
    provenance: list[FactProvenance] = Field(default_factory=list)


class IngestIRPlan(BaseModel):
    """Canonical ingest IR lowered from semantic facts."""

    subject_name: str
    source_language: str = "python"
    operations: list[OperationSpec] = Field(default_factory=list)
    state_slots: list[StateSlotSpec] = Field(default_factory=list)
    artifacts: list[OutputBindingSpec] = Field(default_factory=list)
    edges: list[OperationEdge] = Field(default_factory=list)
    unknowns: list[UnknownFact] = Field(default_factory=list)
    lowering_version: str = "phase2_v1"


class DecompositionDecision(BaseModel):
    """Planner decision for one canonical IR operation."""

    operation_id: str
    decision: str = "keep_atomic"
    planner_source: str = "deterministic"
    reason: str = ""
    evidence: list[str] = Field(default_factory=list)
    child_group_ids: list[str] = Field(default_factory=list)


class PlannedOperationGroup(BaseModel):
    """A planned group of one or more canonical operations."""

    group_id: str
    display_name: str
    group_role: str = "unknown"
    member_operation_ids: list[str] = Field(default_factory=list)
    required_state_slots: list[str] = Field(default_factory=list)
    emitted_outputs: list[OutputBindingSpec] = Field(default_factory=list)
    planner_source: str = "deterministic"
    provenance: list[FactProvenance] = Field(default_factory=list)


class IngestPlanGraph(BaseModel):
    """Planner output over the canonical ingest IR."""

    operation_decisions: list[DecompositionDecision] = Field(default_factory=list)
    planned_groups: list[PlannedOperationGroup] = Field(default_factory=list)
    blocked_operations: list[str] = Field(default_factory=list)
    planner_version: str = "phase3_v1"


class MethodFact(BaseModel):
    """Extracted facts about a single method."""

    name: str
    params: list[str] = Field(default_factory=list)
    return_type: str = ""
    docstring: str = ""
    reads: list[str] = Field(default_factory=list)
    writes: list[str] = Field(default_factory=list)
    calls: list[str] = Field(default_factory=list)
    config_branches: list[ConfigBranch] = Field(default_factory=list)
    source_code: str = ""
    decorators: list[str] = Field(default_factory=list)
    is_opaque: bool = False
    is_external: bool = False
    signature: list[ParameterFact] = Field(default_factory=list)
    return_facts: list[ReturnFact] = Field(default_factory=list)
    call_facts: list[CallFact] = Field(default_factory=list)
    unknown_facts: list[UnknownFact] = Field(default_factory=list)
    semantic_role: str = ""
    config_attributes: list[str] = Field(default_factory=list)
    fitted_attributes: list[str] = Field(default_factory=list)
    provenance: list[FactProvenance] = Field(default_factory=list)
    # Bayesian metadata flags (set by tree-sitter extractors)
    is_oracle: bool = False  # Implements a stateless log-density/gradient target
    is_conjugate: bool = False  # Implements an analytical conjugate update


class OracleEdge(BaseModel):
    """An explicit dependency on a stateless oracle (log-density/gradient).

    Extracted from C++ functional APIs like kthohr/mcmc where a kernel
    function pointer is passed to an algorithm entry point.
    """

    caller: str  # Function or algorithm that consumes the oracle
    oracle_ref: str  # Name/identifier of the oracle function/pointer
    call_site: str = ""  # Optional: the call expression text


class RawDataFlowGraph(BaseModel):
    """Phase 1 output: deterministic extraction of a class's data flow."""

    class_name: str
    source_code: str = ""
    methods: list[MethodFact] = Field(default_factory=list)
    all_attributes: dict[str, list[str]] = Field(default_factory=dict)
    config_branches: list[ConfigBranch] = Field(default_factory=list)
    init_chain: list[str] = Field(default_factory=list)
    cross_window_attrs: list[str] = Field(default_factory=list)
    internal_call_graph: dict[str, list[str]] = Field(default_factory=dict)
    inferred_edges: list[DependencyEdge] = Field(default_factory=list)
    source_language: str = "python"
    is_opaque: bool = False
    is_external: bool = False
    opaque_base_classes: list[str] = Field(default_factory=list)
    attribute_facts: list[AttributeSemanticFact] = Field(default_factory=list)
    config_attributes: list[str] = Field(default_factory=list)
    fitted_attributes: list[str] = Field(default_factory=list)
    derived_attributes: list[str] = Field(default_factory=list)
    semantic_unknowns: list[UnknownFact] = Field(default_factory=list)
    semantic_fact_version: str = "phase1_v1"

    # Bayesian / probabilistic metadata (populated by tree-sitter extractors)
    static_shape: dict[str, str] = Field(
        default_factory=dict,
        description="Compile-time static matrix dimensions, e.g. {'N': '6', 'M': '3'} "
        "from nalgebra SMatrix<f64, N, M>",
    )
    requires_logdet_jacobian: bool = Field(
        default=False,
        description="True when constrained variables use Bijectors.jl transforms",
    )
    cartesian_product_fields: list[list[str]] = Field(
        default_factory=list,
        description="Groups of struct fields forming Cartesian type products "
        "(e.g. [['metric', 'integrator', 'trajectory_sampler']]). "
        "Each group should be emitted as distinct swappable subgraph inputs.",
    )
    oracle_edges: list[OracleEdge] = Field(
        default_factory=list,
        description="Explicit oracle dependencies extracted from functional MCMC APIs",
    )


# ---------------------------------------------------------------------------
# Phase 2: Semantic chunking outputs
# ---------------------------------------------------------------------------


class ConceptualProfile(BaseModel):
    """Domain-agnostic conceptual abstraction of an atom.

    Generated by the Conceptual Abstraction Agent to maximize cross-domain
    discoverability via semantic vector search.
    """

    abstract_name: str = ""
    conceptual_transform: str = ""
    abstract_inputs: list[str] = Field(default_factory=list)
    abstract_outputs: list[str] = Field(default_factory=list)
    algorithmic_properties: list[str] = Field(default_factory=list)
    cross_disciplinary_applications: list[str] = Field(default_factory=list)


class MacroAtomSpec(BaseModel):
    """Specification for one macro-atom produced by the LLM chunker."""

    name: str
    description: str = ""
    method_names: list[str] = Field(default_factory=list)
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    config_params: list[str] = Field(default_factory=list)
    concept_type: ConceptType = ConceptType.CUSTOM
    decorators: list[str] = Field(default_factory=list)
    is_optional: bool = False
    is_opaque: bool = False
    is_external: bool = False
    is_stochastic: bool = Field(
        default=False,
        description="True when atom performs stochastic sampling/transitions.",
    )
    requires_rng_key: bool = Field(
        default=False,
        description="True when atom expects an RNG key/seed input.",
    )
    requires_autodiff: bool = Field(
        default=False,
        description="True when atom depends on automatic differentiation.",
    )
    autodiff_backend: str = Field(
        default="",
        description="Optional AD backend/runtime, e.g. 'jax' or 'LogDensityProblems.jl'.",
    )
    conceptual_profile: ConceptualProfile | None = Field(
        default=None,
        description="Domain-agnostic abstraction for cross-domain retrieval",
    )
    children: list["MacroAtomSpec"] = Field(
        default_factory=list,
        description="Sub-atoms from recursive decomposition",
    )
    sub_edges: list[DependencyEdge] = Field(
        default_factory=list,
        description="DATA_FLOW edges between children from recursive decomposition",
    )
    depth: int = Field(
        default=0,
        description="Depth in the decomposition tree (0 = top-level)",
    )
    source_lines: int = Field(
        default=0,
        description="Line count of underlying method source (for complexity heuristic)",
    )


class StochasticTraceSpec(BaseModel):
    """Specification for stochastic state persisted across atom executions.

    Captures the metadata needed to thread RNG keys and MCMC chain state
    through a Bayesian pipeline without breaking functional purity.
    """

    rng_field: str = Field(
        default="rng_key",
        description="State field name for the RNG key/seed",
    )
    rng_type: str = Field(
        default="jax.random.PRNGKey",
        description="Type annotation for the RNG field",
    )
    trace_field: str = Field(
        default="",
        description="State field name for the MCMC trace (empty if not MCMC)",
    )
    trace_param_dims: tuple[int, ...] = Field(
        default=(),
        description="Parameter dimensions for the trace, e.g. (3,) for 3D",
    )
    chain_count: int = Field(
        default=1,
        ge=1,
        description="Number of parallel MCMC chains",
    )
    warmup_steps: int = Field(
        default=0,
        ge=0,
        description="Number of warmup/burn-in steps",
    )


class StateModelSpec(BaseModel):
    """Specification for a Pydantic state model (cross-window state)."""

    model_name: str
    fields: list[tuple[str, str]] = Field(default_factory=list)
    source_attrs: list[str] = Field(default_factory=list)
    docstring: str = ""
    stochastic: StochasticTraceSpec | None = Field(
        default=None,
        description="Stochastic trace spec for Bayesian state models",
    )


class SubAtomRef(BaseModel):
    """Reference to an existing catalog atom for zoom-in decomposition."""

    atom_name: str
    similarity_score: float = 0.0


class ProposedMacroPlan(BaseModel):
    """LLM's grouping proposal from the semantic chunker."""

    macro_atoms: list[MacroAtomSpec] = Field(default_factory=list)
    state_models: list[StateModelSpec] = Field(default_factory=list)
    sub_atom_refs: list[SubAtomRef] = Field(default_factory=list)
    edge_definitions: list[DependencyEdge] = Field(default_factory=list)
    canonical_ir: IngestIRPlan | None = None
    planning_graph: IngestPlanGraph | None = None


class ValidatedMacroPlan(BaseModel):
    """Plan after critic validation pass."""

    plan: ProposedMacroPlan
    all_attrs_accounted: bool = False
    coverage_report: str = ""
    ir_validated: bool = False
    ir_coverage_report: str = ""


# ---------------------------------------------------------------------------
# Compatibility export helpers
# ---------------------------------------------------------------------------


def canonical_operation_id(name: str) -> str:
    """Normalize a display name into the canonical operation id shape."""
    return name.lower().replace(" ", "_").replace("-", "_")


def legacy_outputs_from_operation(operation: OperationSpec) -> list[IOSpec]:
    """Lower canonical outputs into the legacy IOSpec view."""
    outputs: list[IOSpec] = []
    for binding in operation.emitted_outputs:
        if binding.binding_kind == "self_return":
            continue
        outputs.append(IOSpec(name=binding.output_name, type_desc=binding.type_desc))
    return outputs


def legacy_state_models_from_ir(
    ir: IngestIRPlan,
    existing_state_models: list[StateModelSpec],
) -> list[StateModelSpec]:
    """Build compatibility state-model exports from canonical state slots."""
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
            model_name=f"{ir.subject_name}State",
            fields=[(slot.slot_name, slot.type_desc or "Any") for slot in slots],
            source_attrs=[slot.slot_name for slot in slots],
            docstring=f"Legacy adapter state model for {ir.subject_name}.",
        )
    ]


def legacy_edges_from_ir(ir: IngestIRPlan) -> list[DependencyEdge]:
    """Build compatibility dependency edges from canonical IR edges."""
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


def materialize_legacy_plan_views(plan: ProposedMacroPlan) -> ProposedMacroPlan:
    """Explicitly materialize compatibility views from canonical runtime state.

    The canonical IR and planning graph remain the runtime source of truth.
    This helper exists only to populate legacy-compatible exports for existing
    emitters, tests, and bundle surfaces that still expect macro-atoms or state
    models.
    """
    ir = plan.canonical_ir
    if ir is None or not ir.operations:
        return plan

    by_op_id = {canonical_operation_id(atom.name): atom for atom in plan.macro_atoms}
    macro_atoms: list[MacroAtomSpec] = []
    for operation in ir.operations:
        seed = by_op_id.get(operation.operation_id)
        macro_atoms.append(
            MacroAtomSpec(
                name=seed.name if seed is not None else operation.display_name,
                description=seed.description if seed is not None else operation.display_name,
                method_names=[binding.method_name for binding in operation.method_bindings],
                inputs=(
                    list(seed.inputs)
                    if seed is not None and seed.inputs
                    else list(operation.direct_inputs)
                ),
                outputs=(
                    list(seed.outputs)
                    if seed is not None and seed.outputs
                    else legacy_outputs_from_operation(operation)
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

    return plan.model_copy(
        update={
            "macro_atoms": macro_atoms,
            "state_models": legacy_state_models_from_ir(ir, plan.state_models),
            "edge_definitions": (
                list(plan.edge_definitions)
                if plan.edge_definitions
                else legacy_edges_from_ir(ir)
            ),
        }
    )


# ---------------------------------------------------------------------------
# Final output
# ---------------------------------------------------------------------------


class IngestionBundle(BaseModel):
    """Final output of the ingester pipeline.

    ``cdg`` and ``match_results`` use the exact same types that Round 3
    (Synthesizer) expects, so the Synthesizer can consume ingester output
    directly.
    """

    cdg: CDGExport
    sub_graphs: dict[str, CDGExport] = Field(default_factory=dict)
    generated_atoms: str = ""
    generated_state_models: str = ""
    generated_witnesses: str = ""
    match_results: list[MatchResult] = Field(default_factory=list)
    mypy_passed: bool = False
    ghost_sim_passed: bool = False
    ghost_sim_report: dict = Field(default_factory=dict)

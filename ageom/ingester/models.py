"""Pydantic models for the Smart Ingester (Round 0).

Converts existing stateful Python classes into stateless atom graphs
compatible with the AGEO framework.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from ageom.architect.handoff import CDGExport
from ageom.architect.models import ConceptType, DependencyEdge, IOSpec
from ageom.types import MatchResult


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
    is_external: bool = False


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
    is_external: bool = False
    opaque_base_classes: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Phase 2: Semantic chunking outputs
# ---------------------------------------------------------------------------


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
    is_external: bool = False


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
        default=1, ge=1,
        description="Number of parallel MCMC chains",
    )
    warmup_steps: int = Field(
        default=0, ge=0,
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


class ValidatedMacroPlan(BaseModel):
    """Plan after critic validation pass."""

    plan: ProposedMacroPlan
    all_attrs_accounted: bool = False
    coverage_report: str = ""


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

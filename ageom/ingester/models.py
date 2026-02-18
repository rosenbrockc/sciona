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
    is_optional: bool = False


class StateModelSpec(BaseModel):
    """Specification for a Pydantic state model (cross-window state)."""

    model_name: str
    fields: list[tuple[str, str]] = Field(default_factory=list)
    source_attrs: list[str] = Field(default_factory=list)
    docstring: str = ""


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

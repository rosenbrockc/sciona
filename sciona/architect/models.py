"""Core Pydantic models for the Conceptual Dependency Graph (CDG)."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ConceptType(str, Enum):
    """Algorithmic paradigm categories."""

    SORTING = "sorting"
    SEARCHING = "searching"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GRAPH_TRAVERSAL = "graph_traversal"
    GRAPH_OPTIMIZATION = "graph_optimization"
    STRING_MATCHING = "string_matching"
    GEOMETRY = "geometry"
    ARITHMETIC = "arithmetic"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    ALGEBRA = "algebra"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    SET_THEORY = "set_theory"
    SIGNAL_TRANSFORM = "signal_transform"
    SIGNAL_FILTER = "signal_filter"
    GRAPH_SIGNAL_PROCESSING = "graph_signal_processing"
    NEURAL_NETWORK = "neural_network"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ODE_SOLVER = "ode_solver"
    QUADRATURE = "quadrature"
    RANDOMIZED = "randomized"
    # Bayesian / probabilistic inference
    SAMPLER = "sampler"
    LOG_PROB = "log_prob"
    POSTERIOR_UPDATE = "posterior_update"
    VARIATIONAL_INFERENCE = "variational_inference"
    PRIOR_INIT = "prior_init"
    PRIOR_DISTRIBUTION = "prior_distribution"
    LIKELIHOOD_EVALUATION = "likelihood_evaluation"
    PROBABILISTIC_ORACLE = "probabilistic_oracle"
    ORACLE_GRADIENT = "oracle_gradient"
    MCMC_KERNEL = "mcmc_kernel"
    MCMC_PROPOSAL = "mcmc_proposal"
    VI_ELBO = "vi_elbo"
    SEQUENTIAL_FILTER = "sequential_filter"
    SMC_REWEIGHT = "smc_reweight"
    MESSAGE_PASSING = "message_passing"
    CONJUGATE_UPDATE = "conjugate_update"
    # Data flow / orchestration
    STATE_INIT = "state_init"
    DATA_ASSEMBLY = "data_assembly"
    CONDITIONAL_ROUTING = "conditional_routing"
    DATA_EXTRACTION = "data_extraction"
    # Presentation / observability
    VISUALIZATION = "visualization"
    OBSERVABILITY = "observability"
    CUSTOM = "custom"
    EXTERNAL_TOOL = "external_tool"


class IOSpec(BaseModel):
    """Type specification for a node's inputs/outputs."""

    name: str
    type_desc: str  # e.g., "list[int]", "Graph", "nat -> nat -> Prop"
    constraints: str = ""  # e.g., "sorted", "non-empty", "n > 0"
    required: bool = True
    default_value_repr: str = ""


class NodeStatus(str, Enum):
    """Status of a CDG node through the decomposition process."""

    PENDING = "pending"  # Not yet decomposed
    DECOMPOSED = "decomposed"  # Has children
    ATOMIC = "atomic"  # Leaf — maps to a known primitive
    REJECTED = "rejected"  # Critic rejected this decomposition
    HIGH_RISK = "high_risk"  # Requires novel proof, flagged by Critic
    BLOCKED = "blocked"  # Decomposition terminated without a valid handoff


class AlgorithmicNode(BaseModel):
    """A node in the Conceptual Dependency Graph."""

    node_id: str
    parent_id: str | None = None
    name: str  # e.g., "Sort the List"
    description: str  # Natural language spec
    concept_type: ConceptType
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    children: list[str] = Field(default_factory=list)  # child node_ids
    depth: int = 0
    type_signature: str = ""  # Formal type sig for Round 2 handoff
    matched_primitive: str | None = None  # e.g., "Nat.add_comm" or "heapsort"
    primitive_binding_confidence: float = 0.0
    primitive_binding_source: str = ""
    is_optional: bool = False  # Config-gated branches
    is_opaque: bool = False  # DL boundary: skip internal decomposition
    is_external: bool = False  # External tool call
    parallelizable: bool = False  # Supports parallel execution (e.g., particle swarms)
    conceptual_summary: str = ""
    critic_notes: str = ""
    decomposition_rationale: str = ""


class DependencyEdge(BaseModel):
    """A data-flow edge between CDG nodes."""

    source_id: str
    target_id: str
    output_name: str  # which output of source
    input_name: str  # which input of target
    source_type: str  # type of the data flowing
    target_type: str  # expected type at target
    requires_glue: bool = False  # True if types don't match


class SkeletonGraph(BaseModel):
    """A pre-fabricated graph template for an algorithmic paradigm."""

    paradigm: ConceptType
    name: str  # e.g., "Divide and Conquer"
    description: str
    template_nodes: list[AlgorithmicNode] = Field(default_factory=list)
    template_edges: list[DependencyEdge] = Field(default_factory=list)
    variants: list[str] = Field(
        default_factory=list
    )  # e.g., ["merge_sort", "quicksort"]


class ParamStatus(str, Enum):
    """Audit status of a primitive's tunable parameters."""

    APPROVED = "approved"
    FIXED = "fixed"
    BLOCKED = "blocked"
    DEPRECATED = "deprecated"


class PrimitiveParamSpec(BaseModel):
    """Schema for a single tunable parameter on a primitive."""

    name: str
    kind: Literal["int", "float", "categorical", "bool"]
    default: float | int | str | bool
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    log_scale: bool = False
    choices: list[str | int | float] | None = None
    constraints: str = ""
    semantic_role: str = ""
    safe_to_optimize: bool = True

    # Provenance
    range_source: str = ""
    source_reference: str = ""
    source_confidence: str = ""

    @model_validator(mode="after")
    def _validate_range(self) -> "PrimitiveParamSpec":
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
            )
        return self


class AlgorithmicPrimitive(BaseModel):
    """A known atomic operation from CLRS or a library."""

    name: str
    source: str  # "clrs-30", "coq-100-theorems", "mathlib"
    category: ConceptType
    description: str
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    type_signature: str = ""  # Formal type for Round 2
    clrs_spec: dict = Field(default_factory=dict)  # Raw CLRS spec if from CLRS-30
    uncertainty_factor: float | None = None
    uncertainty_confidence: float = 0.0
    uncertainty_mode: str = ""
    tunable_params: list[PrimitiveParamSpec] = Field(default_factory=list)
    param_status: ParamStatus = ParamStatus.FIXED

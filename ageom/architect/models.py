"""Core Pydantic models for the Conceptual Dependency Graph (CDG)."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


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
    ANALYSIS = "analysis"
    SET_THEORY = "set_theory"
    CUSTOM = "custom"


class IOSpec(BaseModel):
    """Type specification for a node's inputs/outputs."""

    name: str
    type_desc: str  # e.g., "list[int]", "Graph", "nat -> nat -> Prop"
    constraints: str = ""  # e.g., "sorted", "non-empty", "n > 0"


class NodeStatus(str, Enum):
    """Status of a CDG node through the decomposition process."""

    PENDING = "pending"  # Not yet decomposed
    DECOMPOSED = "decomposed"  # Has children
    ATOMIC = "atomic"  # Leaf — maps to a known primitive
    REJECTED = "rejected"  # Critic rejected this decomposition
    HIGH_RISK = "high_risk"  # Requires novel proof, flagged by Critic


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
    variants: list[str] = Field(default_factory=list)  # e.g., ["merge_sort", "quicksort"]


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

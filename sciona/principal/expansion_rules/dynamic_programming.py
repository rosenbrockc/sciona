"""Expansion rules for the Dynamic Programming family.

Defines DPO rules and diagnostic functions that let the expansion engine
insert sparsity detection, constraint pruning, table compression, and
subproblem overlap validation into DP CDGs.

DP skeleton topology (5 nodes, linear pipeline):

    Define Subproblems → Base Case → Recurrence → Memoize → Extract Solution

Expansion insertion points:
  - Before Recurrence: sparsity detection, constraint pruning
  - After Memoize: table compression
  - Before Base Case: subproblem overlap validation

All diagnostics are pure functions of DP table intermediates.
"""

from __future__ import annotations

import logging

import numpy as np

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)

logger = logging.getLogger(__name__)

_DOMAIN = "dynamic_programming"

# DP skeleton node names
_DEFINE_SUBPROBLEMS = "Define Subproblems"
_BASE_CASE = "Base Case"
_RECURRENCE = "Recurrence"
_MEMOIZE = "Memoize"
_EXTRACT_SOLUTION = "Extract Solution"


# ---------------------------------------------------------------------------
# Node / edge helpers
# ---------------------------------------------------------------------------


def _node(
    node_id: str,
    name: str,
    concept_type: ConceptType,
    *,
    matched_primitive: str | None = None,
    inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None,
    description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id,
        name=name,
        description=description or name,
        concept_type=concept_type,
        status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive,
        inputs=inputs or [],
        outputs=outputs or [],
        type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str,
    target_id: str,
    output_name: str = "out",
    input_name: str = "in",
    type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id,
        target_id=target_id,
        output_name=output_name,
        input_name=input_name,
        source_type=type_desc,
        target_type=type_desc,
    )


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_sparsity_detection() -> RewriteRule:
    """Interpose ``detect_table_sparsity`` before Recurrence.

    When a large fraction of the DP table is unused, switching to a
    sparse representation can drastically reduce memory usage.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    recurrence = _node(
        "recurrence",
        _RECURRENCE,
        ConceptType.DYNAMIC_PROGRAMMING,
    )
    lhs = CDGExport(nodes=[src, recurrence], edges=[_edge("src", "recurrence")])
    interface = CDGExport(nodes=[src, recurrence], edges=[])

    sparsity = _node(
        "sparsity",
        "Detect Table Sparsity",
        ConceptType.DYNAMIC_PROGRAMMING,
        matched_primitive="detect_table_sparsity",
        inputs=[
            IOSpec(name="table", type_desc="ndarray"),
            IOSpec(name="fill_mask", type_desc="Optional[ndarray]"),
        ],
        outputs=[
            IOSpec(name="density", type_desc="float"),
            IOSpec(name="sparse_indices", type_desc="ndarray"),
        ],
        description="Compute fraction of DP table cells that are actually filled/used.",
        type_signature="ndarray, Optional[ndarray] -> tuple[float, ndarray]",
    )
    rhs = CDGExport(
        nodes=[src, sparsity, recurrence],
        edges=[
            _edge("src", "sparsity"),
            _edge("sparsity", "recurrence"),
        ],
    )

    return RewriteRule(
        name="insert_sparsity_detection_before_recurrence",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "recurrence": "recurrence"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "recurrence": "recurrence"}, edge_map={}),
        priority=2,
    )


def _build_insert_constraint_pruning() -> RewriteRule:
    """Interpose ``prune_infeasible_states`` before Recurrence.

    When more than half the DP state space is infeasible, pruning
    those states avoids wasting computation on unreachable cells.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    recurrence = _node(
        "recurrence",
        _RECURRENCE,
        ConceptType.DYNAMIC_PROGRAMMING,
    )
    lhs = CDGExport(nodes=[src, recurrence], edges=[_edge("src", "recurrence")])
    interface = CDGExport(nodes=[src, recurrence], edges=[])

    pruning = _node(
        "pruning",
        "Prune Infeasible States",
        ConceptType.DYNAMIC_PROGRAMMING,
        matched_primitive="prune_infeasible_states",
        inputs=[
            IOSpec(name="table_shape", type_desc="tuple[int, ...]"),
            IOSpec(name="constraints", type_desc="ndarray"),
            IOSpec(name="state_bounds", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="feasible_mask", type_desc="ndarray"),
            IOSpec(name="n_pruned", type_desc="int"),
        ],
        description="Build a feasibility mask over the DP state space given bound constraints.",
        type_signature="tuple[int, ...], ndarray, ndarray -> tuple[ndarray, int]",
    )
    rhs = CDGExport(
        nodes=[src, pruning, recurrence],
        edges=[
            _edge("src", "pruning"),
            _edge("pruning", "recurrence"),
        ],
    )

    return RewriteRule(
        name="insert_constraint_pruning_before_recurrence",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "recurrence": "recurrence"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "recurrence": "recurrence"}, edge_map={}),
        priority=3,
    )


def _build_insert_table_compression() -> RewriteRule:
    """Interpose ``compress_dp_table`` after Memoize.

    When the reuse distance is bounded (e.g. only the last 1–2 rows are
    needed), older rows can be discarded to save memory.
    """
    memoize = _node(
        "memoize",
        _MEMOIZE,
        ConceptType.DYNAMIC_PROGRAMMING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[memoize, sink], edges=[_edge("memoize", "sink")])
    interface = CDGExport(nodes=[memoize, sink], edges=[])

    compress = _node(
        "compress",
        "Compress DP Table",
        ConceptType.DYNAMIC_PROGRAMMING,
        matched_primitive="compress_dp_table",
        inputs=[
            IOSpec(name="table", type_desc="ndarray"),
            IOSpec(name="reuse_distance", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="compressed_table", type_desc="ndarray"),
            IOSpec(name="memory_saved_ratio", type_desc="float"),
        ],
        description="Retain only the most recent reuse_distance rows of a DP table.",
        type_signature="ndarray, int -> tuple[ndarray, float]",
    )
    rhs = CDGExport(
        nodes=[memoize, compress, sink],
        edges=[
            _edge("memoize", "compress"),
            _edge("compress", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_table_compression_after_memoize",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"memoize": "memoize", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"memoize": "memoize", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _build_insert_overlap_validation() -> RewriteRule:
    """Interpose ``validate_subproblem_overlap`` before Base Case.

    When subproblem reuse is low, the problem may not actually benefit
    from memoization and divide-and-conquer could be more appropriate.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    base_case = _node(
        "base_case",
        _BASE_CASE,
        ConceptType.DYNAMIC_PROGRAMMING,
    )
    lhs = CDGExport(nodes=[src, base_case], edges=[_edge("src", "base_case")])
    interface = CDGExport(nodes=[src, base_case], edges=[])

    overlap = _node(
        "overlap",
        "Validate Subproblem Overlap",
        ConceptType.DYNAMIC_PROGRAMMING,
        matched_primitive="validate_subproblem_overlap",
        inputs=[
            IOSpec(name="call_counts", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="reuse_ratio", type_desc="float"),
            IOSpec(name="has_overlap", type_desc="bool"),
        ],
        description="Check whether subproblems are reused enough to justify memoization.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, overlap, base_case],
        edges=[
            _edge("src", "overlap"),
            _edge("overlap", "base_case"),
        ],
    )

    return RewriteRule(
        name="insert_overlap_validation_before_base_case",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "base_case": "base_case"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "base_case": "base_case"}, edge_map={}),
        priority=2,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_table_sparsity(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect sparse DP tables (< 30% filled)."""
    intermediates = context.intermediates or {}
    table_density = intermediates.get("table_density")

    if table_density is None:
        return None

    try:
        density = float(table_density)
    except (ValueError, TypeError):
        return None

    if density < 0.3:
        return ExpansionDiagnostic(
            rule_name="insert_sparsity_detection_before_recurrence",
            severity=min(1.0, (0.3 - density) / 0.3),
            evidence=(
                f"DP table density {density:.2f} is below 0.3 threshold "
                f"— sparse representation may save memory"
            ),
            metric_name="table_density",
            metric_value=density,
            threshold=0.3,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_infeasible_states(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect high infeasible state fraction (> 50%)."""
    intermediates = context.intermediates or {}
    infeasible_fraction = intermediates.get("infeasible_fraction")

    if infeasible_fraction is None:
        return None

    try:
        fraction = float(infeasible_fraction)
    except (ValueError, TypeError):
        return None

    if fraction > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_constraint_pruning_before_recurrence",
            severity=min(1.0, (fraction - 0.5) / 0.5),
            evidence=(
                f"Infeasible state fraction {fraction:.2f} exceeds 0.5 threshold "
                f"— pruning can skip {fraction * 100:.0f}% of state space"
            ),
            metric_name="infeasible_fraction",
            metric_value=fraction,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_table_memory(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect large DP tables with bounded reuse distance."""
    intermediates = context.intermediates or {}
    table_memory_mb = intermediates.get("table_memory_mb")
    reuse_distance = intermediates.get("reuse_distance")

    if table_memory_mb is None or reuse_distance is None:
        return None

    try:
        mem_mb = float(table_memory_mb)
        rd = int(reuse_distance)
    except (ValueError, TypeError):
        return None

    if mem_mb > 100 and rd > 0:
        return ExpansionDiagnostic(
            rule_name="insert_table_compression_after_memoize",
            severity=min(1.0, mem_mb / 1000.0),
            evidence=(
                f"DP table uses {mem_mb:.1f} MB with reuse distance {rd} "
                f"— compression can discard older rows"
            ),
            metric_name="table_memory_mb",
            metric_value=mem_mb,
            threshold=100.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_subproblem_overlap(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect low subproblem reuse (< 1.5 ratio)."""
    intermediates = context.intermediates or {}
    call_counts = intermediates.get("call_counts")

    if call_counts is None:
        return None

    try:
        counts = np.asarray(call_counts, dtype=np.float64)
    except (ValueError, TypeError):
        return None

    active = counts[counts > 0]
    if len(active) == 0:
        return None

    reuse_ratio = float(np.mean(active))

    if reuse_ratio < 1.5:
        return ExpansionDiagnostic(
            rule_name="insert_overlap_validation_before_base_case",
            severity=min(1.0, (1.5 - reuse_ratio) / 1.5),
            evidence=(
                f"Subproblem reuse ratio {reuse_ratio:.2f} is below 1.5 threshold "
                f"— divide-and-conquer may be more appropriate than DP"
            ),
            metric_name="reuse_ratio",
            metric_value=reuse_ratio,
            threshold=1.5,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class DynamicProgrammingExpansionRuleSet:
    """Expansion rules for dynamic programming pipelines."""

    name = "dynamic_programming"
    domain = "dynamic_programming"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_sparsity_detection(),
            _build_insert_constraint_pruning(),
            _build_insert_table_compression(),
            _build_insert_overlap_validation(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        sparsity = _diagnose_table_sparsity(cdg, context)
        if sparsity is not None:
            diagnostics.append(sparsity)

        infeasible = _diagnose_infeasible_states(cdg, context)
        if infeasible is not None:
            diagnostics.append(infeasible)

        memory = _diagnose_table_memory(cdg, context)
        if memory is not None:
            diagnostics.append(memory)

        overlap = _diagnose_subproblem_overlap(cdg, context)
        if overlap is not None:
            diagnostics.append(overlap)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

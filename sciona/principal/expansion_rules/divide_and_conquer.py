"""Expansion rules for the Divide and Conquer family (Merge Sort, Quicksort, Strassen, Closest Pair).

Defines DPO rules and diagnostic functions that let the expansion engine
insert split balance analysis, recursion depth monitoring, merge cost
profiling, and subproblem overlap detection into D&C CDGs.

D&C skeleton topology (4 nodes, diamond):

    Split → Recurse Left  → Merge
    Split → Recurse Right → Merge

Expansion insertion points:
  - After Split: split balance measurement
  - Before Recurse Left: recursion depth check
  - After Merge: merge cost profiling
  - Before Split: subproblem overlap detection

All diagnostics are pure functions of D&C intermediates.
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

_DOMAIN = "divide_and_conquer"

# D&C skeleton node names
_SPLIT = "Split"
_RECURSE_LEFT = "Recurse Left"
_RECURSE_RIGHT = "Recurse Right"
_MERGE = "Merge"


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


def _build_insert_split_balance() -> RewriteRule:
    """Interpose ``measure_split_balance`` after Split.

    Unbalanced splits (e.g. quicksort worst-case pivot) degrade D&C
    from O(n log n) to O(n²).  This check measures partition balance
    so the user is warned about pathological inputs.
    """
    split = _node(
        "split",
        _SPLIT,
        ConceptType.DIVIDE_AND_CONQUER,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[split, sink], edges=[_edge("split", "sink")])
    interface = CDGExport(nodes=[split, sink], edges=[])

    balance = _node(
        "balance",
        "Measure Split Balance",
        ConceptType.DIVIDE_AND_CONQUER,
        matched_primitive="measure_split_balance",
        inputs=[
            IOSpec(name="left_sizes", type_desc="ndarray"),
            IOSpec(name="right_sizes", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="mean_balance", type_desc="float"),
            IOSpec(name="per_level_balance", type_desc="ndarray"),
        ],
        description="Measure the balance of divide-and-conquer splits.",
        type_signature="ndarray, ndarray -> tuple[float, ndarray]",
    )
    rhs = CDGExport(
        nodes=[split, balance, sink],
        edges=[
            _edge("split", "balance"),
            _edge("balance", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_split_balance_after_split",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"split": "split", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"split": "split", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_recursion_depth_check() -> RewriteRule:
    """Interpose ``check_recursion_depth`` before Recurse Left.

    Excessive recursion depth suggests unbalanced splits or a missing
    base case, risking stack overflow.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    recurse = _node(
        "recurse",
        _RECURSE_LEFT,
        ConceptType.DIVIDE_AND_CONQUER,
    )
    lhs = CDGExport(nodes=[src, recurse], edges=[_edge("src", "recurse")])
    interface = CDGExport(nodes=[src, recurse], edges=[])

    depth_check = _node(
        "depth_check",
        "Check Recursion Depth",
        ConceptType.DIVIDE_AND_CONQUER,
        matched_primitive="check_recursion_depth",
        inputs=[
            IOSpec(name="actual_depth", type_desc="int"),
            IOSpec(name="input_size", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="depth_ratio", type_desc="float"),
            IOSpec(name="is_excessive", type_desc="bool"),
        ],
        description="Check whether recursion depth is excessive relative to input size.",
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, depth_check, recurse],
        edges=[
            _edge("src", "depth_check"),
            _edge("depth_check", "recurse"),
        ],
    )

    return RewriteRule(
        name="insert_recursion_depth_check_before_recurse",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "recurse": "recurse"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "recurse": "recurse"}, edge_map={}),
        priority=2,
    )


def _build_insert_merge_cost_profiling() -> RewriteRule:
    """Interpose ``profile_merge_cost`` after Merge.

    When merge cost dominates total runtime, optimizing the merge
    step yields the biggest speedup.
    """
    merge = _node(
        "merge",
        _MERGE,
        ConceptType.DIVIDE_AND_CONQUER,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[merge, sink], edges=[_edge("merge", "sink")])
    interface = CDGExport(nodes=[merge, sink], edges=[])

    profiler = _node(
        "profiler",
        "Profile Merge Cost",
        ConceptType.DIVIDE_AND_CONQUER,
        matched_primitive="profile_merge_cost",
        inputs=[
            IOSpec(name="merge_times", type_desc="ndarray"),
            IOSpec(name="total_times", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="mean_merge_fraction", type_desc="float"),
            IOSpec(name="per_level_fraction", type_desc="ndarray"),
        ],
        description="Profile the fraction of total time spent in merge operations.",
        type_signature="ndarray, ndarray -> tuple[float, ndarray]",
    )
    rhs = CDGExport(
        nodes=[merge, profiler, sink],
        edges=[
            _edge("merge", "profiler"),
            _edge("profiler", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_merge_cost_profiling_after_merge",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"merge": "merge", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"merge": "merge", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _build_insert_subproblem_overlap_detection() -> RewriteRule:
    """Interpose ``detect_subproblem_overlap`` before Split.

    When subproblems repeat frequently, pure D&C wastes work
    recomputing the same results — DP / memoization is more appropriate.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    split = _node(
        "split",
        _SPLIT,
        ConceptType.DIVIDE_AND_CONQUER,
    )
    lhs = CDGExport(nodes=[src, split], edges=[_edge("src", "split")])
    interface = CDGExport(nodes=[src, split], edges=[])

    overlap = _node(
        "overlap",
        "Detect Subproblem Overlap",
        ConceptType.DIVIDE_AND_CONQUER,
        matched_primitive="detect_subproblem_overlap",
        inputs=[
            IOSpec(name="subproblem_hashes", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="overlap_ratio", type_desc="float"),
            IOSpec(name="n_duplicates", type_desc="int"),
        ],
        description="Detect repeated subproblems suggesting DP would be more efficient.",
        type_signature="ndarray -> tuple[float, int]",
    )
    rhs = CDGExport(
        nodes=[src, overlap, split],
        edges=[
            _edge("src", "overlap"),
            _edge("overlap", "split"),
        ],
    )

    return RewriteRule(
        name="insert_subproblem_overlap_detection_before_split",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "split": "split"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "split": "split"}, edge_map={}),
        priority=2,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_split_balance(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect unbalanced splits (mean balance < 0.5)."""
    intermediates = context.intermediates or {}
    mean_balance = intermediates.get("split_balance")

    if mean_balance is None:
        return None

    try:
        balance = float(mean_balance)
    except (ValueError, TypeError):
        return None

    if balance < 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_split_balance_after_split",
            severity=min(1.0, (0.5 - balance) / 0.5),
            evidence=(
                f"Mean split balance {balance:.2f} is below 0.5 threshold "
                f"— partitions are highly unbalanced"
            ),
            metric_name="split_balance",
            metric_value=balance,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_recursion_depth(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect excessive recursion depth relative to input size."""
    intermediates = context.intermediates or {}
    actual_depth = intermediates.get("recursion_depth")
    input_size = intermediates.get("input_size")

    if actual_depth is None or input_size is None:
        return None

    try:
        depth = int(actual_depth)
        n = int(input_size)
    except (ValueError, TypeError):
        return None

    if n <= 1:
        return None

    expected_max = 2.0 * np.log2(max(n, 2))
    ratio = depth / expected_max

    if ratio > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_recursion_depth_check_before_recurse",
            severity=min(1.0, (ratio - 1.0) / 2.0),
            evidence=(
                f"Recursion depth {depth} exceeds 2·log₂({n})={expected_max:.1f} "
                f"by {ratio:.1f}x — splits may be unbalanced or base case too small"
            ),
            metric_name="depth_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_merge_cost(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect merge-dominated workloads (merge fraction > 0.5)."""
    intermediates = context.intermediates or {}
    merge_fraction = intermediates.get("merge_fraction")

    if merge_fraction is None:
        return None

    try:
        fraction = float(merge_fraction)
    except (ValueError, TypeError):
        return None

    if fraction > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_merge_cost_profiling_after_merge",
            severity=min(1.0, (fraction - 0.5) / 0.5),
            evidence=(
                f"Merge fraction {fraction:.2f} exceeds 0.5 threshold "
                f"— merge step dominates {fraction * 100:.0f}% of total runtime"
            ),
            metric_name="merge_fraction",
            metric_value=fraction,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_subproblem_overlap(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect high subproblem overlap (ratio > 0.1)."""
    intermediates = context.intermediates or {}
    overlap_ratio = intermediates.get("subproblem_overlap_ratio")

    if overlap_ratio is None:
        return None

    try:
        ratio = float(overlap_ratio)
    except (ValueError, TypeError):
        return None

    if ratio > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_subproblem_overlap_detection_before_split",
            severity=min(1.0, ratio),
            evidence=(
                f"Subproblem overlap ratio {ratio:.2f} exceeds 0.1 threshold "
                f"— consider DP / memoization instead of pure D&C"
            ),
            metric_name="subproblem_overlap_ratio",
            metric_value=ratio,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class DivideAndConquerExpansionRuleSet:
    """Expansion rules for divide-and-conquer pipelines (Merge Sort, Quicksort, Strassen, Closest Pair)."""

    name = "divide_and_conquer"
    domain = "divide_and_conquer"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_split_balance(),
            _build_insert_recursion_depth_check(),
            _build_insert_merge_cost_profiling(),
            _build_insert_subproblem_overlap_detection(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        balance = _diagnose_split_balance(cdg, context)
        if balance is not None:
            diagnostics.append(balance)

        depth = _diagnose_recursion_depth(cdg, context)
        if depth is not None:
            diagnostics.append(depth)

        merge = _diagnose_merge_cost(cdg, context)
        if merge is not None:
            diagnostics.append(merge)

        overlap = _diagnose_subproblem_overlap(cdg, context)
        if overlap is not None:
            diagnostics.append(overlap)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

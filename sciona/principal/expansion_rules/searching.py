"""Expansion rules for the Searching family (Binary, Linear, Interpolation).

Defines DPO rules and diagnostic functions that let the expansion engine
insert sorted order validation, distribution uniformity analysis, midpoint
overflow detection, and iteration count analysis into searching CDGs.

Searching skeleton topology (3 nodes, linear pipeline):

    Init Bounds → Probe → Narrow

Expansion insertion points:
  - Before Init Bounds: sorted order validation, distribution uniformity
  - Before Probe: midpoint overflow detection
  - After Narrow: iteration count analysis

All diagnostics are pure functions of searching intermediates.
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

_DOMAIN = "searching"

# Searching skeleton node names
_INIT_BOUNDS = "Init Bounds"
_PROBE = "Probe"
_NARROW = "Narrow"


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


def _build_insert_sorted_order_validation() -> RewriteRule:
    """Interpose ``validate_sorted_order`` before Init Bounds.

    Binary search and interpolation search require sorted input.
    Unsorted data produces silently incorrect results.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    init = _node(
        "init",
        _INIT_BOUNDS,
        ConceptType.SEARCHING,
    )
    lhs = CDGExport(nodes=[src, init], edges=[_edge("src", "init")])
    interface = CDGExport(nodes=[src, init], edges=[])

    validate = _node(
        "validate",
        "Validate Sorted Order",
        ConceptType.SEARCHING,
        matched_primitive="validate_sorted_order",
        inputs=[
            IOSpec(name="data", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="n_violations", type_desc="int"),
            IOSpec(name="is_sorted", type_desc="bool"),
        ],
        description="Validate that input data is sorted in non-decreasing order.",
        type_signature="ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(
        nodes=[src, validate, init],
        edges=[
            _edge("src", "validate"),
            _edge("validate", "init"),
        ],
    )

    return RewriteRule(
        name="insert_sorted_order_validation_before_init",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        priority=3,
    )


def _build_insert_distribution_uniformity_analysis() -> RewriteRule:
    """Interpose ``analyze_distribution_uniformity`` before Init Bounds.

    Interpolation search is O(log log n) for uniform data but
    degrades to O(n) for skewed distributions.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    init = _node(
        "init",
        _INIT_BOUNDS,
        ConceptType.SEARCHING,
    )
    lhs = CDGExport(nodes=[src, init], edges=[_edge("src", "init")])
    interface = CDGExport(nodes=[src, init], edges=[])

    uniformity = _node(
        "uniformity",
        "Analyze Distribution Uniformity",
        ConceptType.SEARCHING,
        matched_primitive="analyze_distribution_uniformity",
        inputs=[
            IOSpec(name="data", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="uniformity_score", type_desc="float"),
            IOSpec(name="recommendation", type_desc="str"),
        ],
        description="Analyze how uniformly distributed the search data is.",
        type_signature="ndarray -> tuple[float, str]",
    )
    rhs = CDGExport(
        nodes=[src, uniformity, init],
        edges=[
            _edge("src", "uniformity"),
            _edge("uniformity", "init"),
        ],
    )

    return RewriteRule(
        name="insert_distribution_uniformity_analysis_before_init",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "init": "init"}, edge_map={}),
        priority=2,
    )


def _build_insert_midpoint_overflow_detection() -> RewriteRule:
    """Interpose ``detect_midpoint_overflow`` before Probe.

    Naive midpoint calculation (lo + hi) / 2 can overflow for
    large indices.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    probe = _node(
        "probe",
        _PROBE,
        ConceptType.SEARCHING,
    )
    lhs = CDGExport(nodes=[src, probe], edges=[_edge("src", "probe")])
    interface = CDGExport(nodes=[src, probe], edges=[])

    overflow = _node(
        "overflow",
        "Detect Midpoint Overflow",
        ConceptType.SEARCHING,
        matched_primitive="detect_midpoint_overflow",
        inputs=[
            IOSpec(name="lo", type_desc="int"),
            IOSpec(name="hi", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="would_overflow", type_desc="bool"),
            IOSpec(name="safe_mid", type_desc="int"),
        ],
        description="Detect potential integer overflow in midpoint calculation.",
        type_signature="int, int -> tuple[bool, int]",
    )
    rhs = CDGExport(
        nodes=[src, overflow, probe],
        edges=[
            _edge("src", "overflow"),
            _edge("overflow", "probe"),
        ],
    )

    return RewriteRule(
        name="insert_midpoint_overflow_detection_before_probe",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "probe": "probe"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "probe": "probe"}, edge_map={}),
        priority=2,
    )


def _build_insert_iteration_count_analysis() -> RewriteRule:
    """Interpose ``analyze_iteration_count`` after Narrow.

    Excessive iterations indicate a bug or degenerate search case.
    """
    narrow = _node(
        "narrow",
        _NARROW,
        ConceptType.SEARCHING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[narrow, sink], edges=[_edge("narrow", "sink")])
    interface = CDGExport(nodes=[narrow, sink], edges=[])

    iter_analysis = _node(
        "iter_analysis",
        "Analyze Iteration Count",
        ConceptType.SEARCHING,
        matched_primitive="analyze_iteration_count",
        inputs=[
            IOSpec(name="n_iterations", type_desc="int"),
            IOSpec(name="n_elements", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="iteration_ratio", type_desc="float"),
            IOSpec(name="is_excessive", type_desc="bool"),
        ],
        description="Check whether search iteration count is excessive.",
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[narrow, iter_analysis, sink],
        edges=[
            _edge("narrow", "iter_analysis"),
            _edge("iter_analysis", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_iteration_count_analysis_after_narrow",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"narrow": "narrow", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"narrow": "narrow", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_unsorted_input(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect unsorted input data."""
    intermediates = context.intermediates or {}
    n_sort_violations = intermediates.get("sort_violations")

    if n_sort_violations is None:
        return None

    try:
        violations = int(n_sort_violations)
    except (ValueError, TypeError):
        return None

    if violations > 0:
        return ExpansionDiagnostic(
            rule_name="insert_sorted_order_validation_before_init",
            severity=min(1.0, violations / 10.0),
            evidence=(
                f"{violations} sort order violation(s) detected "
                f"— input is not sorted, search results will be incorrect"
            ),
            metric_name="sort_violations",
            metric_value=float(violations),
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_distribution_uniformity(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect non-uniform data distribution for interpolation search."""
    intermediates = context.intermediates or {}
    uniformity_score = intermediates.get("uniformity_score")

    if uniformity_score is None:
        return None

    try:
        score = float(uniformity_score)
    except (ValueError, TypeError):
        return None

    if score < 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_distribution_uniformity_analysis_before_init",
            severity=min(1.0, (0.5 - score) / 0.5),
            evidence=(
                f"Distribution uniformity {score:.2f} is below 0.5 threshold "
                f"— interpolation search may degrade to O(n)"
            ),
            metric_name="uniformity_score",
            metric_value=score,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_midpoint_overflow(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect large index values that risk midpoint overflow."""
    intermediates = context.intermediates or {}
    max_index = intermediates.get("max_index")

    if max_index is None:
        return None

    try:
        idx = int(max_index)
    except (ValueError, TypeError):
        return None

    threshold = np.iinfo(np.int64).max // 2
    if idx > threshold:
        return ExpansionDiagnostic(
            rule_name="insert_midpoint_overflow_detection_before_probe",
            severity=1.0,
            evidence=(
                f"Maximum index {idx} exceeds int64/2 "
                f"— naive midpoint calculation will overflow"
            ),
            metric_name="max_index",
            metric_value=float(idx),
            threshold=float(threshold),
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_iteration_count(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect excessive search iterations."""
    intermediates = context.intermediates or {}
    n_iterations = intermediates.get("search_iterations")
    n_elements = intermediates.get("n_elements")

    if n_iterations is None or n_elements is None:
        return None

    try:
        iters = int(n_iterations)
        n = int(n_elements)
    except (ValueError, TypeError):
        return None

    if n <= 1:
        return None

    expected = 2.0 * np.log2(max(n, 2))
    ratio = iters / expected

    if ratio > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_iteration_count_analysis_after_narrow",
            severity=min(1.0, (ratio - 1.0) / 2.0),
            evidence=(
                f"Iteration ratio {ratio:.2f} exceeds 1.0 "
                f"({iters} iterations vs {expected:.0f} expected for n={n})"
            ),
            metric_name="iteration_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SearchingExpansionRuleSet:
    """Expansion rules for searching pipelines (Binary, Linear, Interpolation)."""

    name = "searching"
    domain = "searching"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_sorted_order_validation(),
            _build_insert_distribution_uniformity_analysis(),
            _build_insert_midpoint_overflow_detection(),
            _build_insert_iteration_count_analysis(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        unsorted = _diagnose_unsorted_input(cdg, context)
        if unsorted is not None:
            diagnostics.append(unsorted)

        uniformity = _diagnose_distribution_uniformity(cdg, context)
        if uniformity is not None:
            diagnostics.append(uniformity)

        overflow = _diagnose_midpoint_overflow(cdg, context)
        if overflow is not None:
            diagnostics.append(overflow)

        iters = _diagnose_iteration_count(cdg, context)
        if iters is not None:
            diagnostics.append(iters)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

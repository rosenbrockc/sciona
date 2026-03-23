"""Expansion rules for the Sorting family (Insertion, Heapsort, Quicksort, Merge Sort).

Defines DPO rules and diagnostic functions that let the expansion engine
insert presortedness detection, comparison count analysis, swap count
analysis, and stability validation into sorting CDGs.

Sorting skeleton topology (3 nodes, linear pipeline):

    Compare → Swap → Recurse/Iterate

Expansion insertion points:
  - Before Compare: presortedness detection
  - After Compare: comparison count analysis
  - After Swap: swap count analysis
  - After Recurse/Iterate: stability validation

All diagnostics are pure functions of sorting intermediates.
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

_DOMAIN = "sorting"

# Sorting skeleton node names
_COMPARE = "Compare"
_SWAP = "Swap"
_RECURSE_ITERATE = "Recurse/Iterate"


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


def _build_insert_presortedness_detection() -> RewriteRule:
    """Interpose ``measure_presortedness`` before Compare.

    Nearly-sorted inputs can be handled more efficiently by adaptive
    algorithms (e.g. Timsort, insertion sort).  This pre-check measures
    existing order to guide algorithm selection.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    compare = _node(
        "compare",
        _COMPARE,
        ConceptType.SORTING,
    )
    lhs = CDGExport(nodes=[src, compare], edges=[_edge("src", "compare")])
    interface = CDGExport(nodes=[src, compare], edges=[])

    presort = _node(
        "presort",
        "Measure Presortedness",
        ConceptType.SORTING,
        matched_primitive="measure_presortedness",
        inputs=[
            IOSpec(name="data", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="disorder_ratio", type_desc="float"),
            IOSpec(name="n_adjacent_inversions", type_desc="int"),
        ],
        description="Measure how sorted the input already is.",
        type_signature="ndarray -> tuple[float, int]",
    )
    rhs = CDGExport(
        nodes=[src, presort, compare],
        edges=[
            _edge("src", "presort"),
            _edge("presort", "compare"),
        ],
    )

    return RewriteRule(
        name="insert_presortedness_detection_before_compare",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "compare": "compare"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "compare": "compare"}, edge_map={}),
        priority=3,
    )


def _build_insert_comparison_count_analysis() -> RewriteRule:
    """Interpose ``analyze_comparison_count`` after Compare.

    Excessive comparisons indicate degenerate behavior (e.g. quicksort
    hitting O(n²) on already-sorted input with bad pivot selection).
    """
    compare = _node(
        "compare",
        _COMPARE,
        ConceptType.SORTING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[compare, sink], edges=[_edge("compare", "sink")])
    interface = CDGExport(nodes=[compare, sink], edges=[])

    comp_analysis = _node(
        "comp_analysis",
        "Analyze Comparison Count",
        ConceptType.SORTING,
        matched_primitive="analyze_comparison_count",
        inputs=[
            IOSpec(name="n_comparisons", type_desc="int"),
            IOSpec(name="n_elements", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="comparison_ratio", type_desc="float"),
            IOSpec(name="is_excessive", type_desc="bool"),
        ],
        description="Check whether comparison count is excessive.",
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[compare, comp_analysis, sink],
        edges=[
            _edge("compare", "comp_analysis"),
            _edge("comp_analysis", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_comparison_count_analysis_after_compare",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"compare": "compare", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"compare": "compare", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_swap_count_analysis() -> RewriteRule:
    """Interpose ``analyze_swap_count`` after Swap.

    Excessive swaps indicate high data movement cost, suggesting
    an algorithm with fewer moves (e.g. merge sort over insertion sort).
    """
    swap = _node(
        "swap",
        _SWAP,
        ConceptType.SORTING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[swap, sink], edges=[_edge("swap", "sink")])
    interface = CDGExport(nodes=[swap, sink], edges=[])

    swap_analysis = _node(
        "swap_analysis",
        "Analyze Swap Count",
        ConceptType.SORTING,
        matched_primitive="analyze_swap_count",
        inputs=[
            IOSpec(name="n_swaps", type_desc="int"),
            IOSpec(name="n_elements", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="swap_ratio", type_desc="float"),
            IOSpec(name="is_excessive", type_desc="bool"),
        ],
        description="Check whether swap/move count is excessive.",
        type_signature="int, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[swap, swap_analysis, sink],
        edges=[
            _edge("swap", "swap_analysis"),
            _edge("swap_analysis", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_swap_count_analysis_after_swap",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"swap": "swap", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"swap": "swap", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_stability_validation() -> RewriteRule:
    """Interpose ``validate_stability`` after Recurse/Iterate.

    When equal-key order matters (e.g. multi-key sort), an unstable
    algorithm silently corrupts secondary ordering.
    """
    recurse = _node(
        "recurse",
        _RECURSE_ITERATE,
        ConceptType.SORTING,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[recurse, sink], edges=[_edge("recurse", "sink")])
    interface = CDGExport(nodes=[recurse, sink], edges=[])

    stability = _node(
        "stability",
        "Validate Stability",
        ConceptType.SORTING,
        matched_primitive="validate_stability",
        inputs=[
            IOSpec(name="keys", type_desc="ndarray"),
            IOSpec(name="original_indices", type_desc="ndarray"),
            IOSpec(name="sorted_indices", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="n_violations", type_desc="int"),
            IOSpec(name="is_stable", type_desc="bool"),
        ],
        description="Check whether a sort preserves the relative order of equal keys.",
        type_signature="ndarray, ndarray, ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(
        nodes=[recurse, stability, sink],
        edges=[
            _edge("recurse", "stability"),
            _edge("stability", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_stability_validation_after_recurse",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"recurse": "recurse", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"recurse": "recurse", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_presortedness(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect nearly-sorted input (disorder ratio < 0.1)."""
    intermediates = context.intermediates or {}
    disorder_ratio = intermediates.get("disorder_ratio")

    if disorder_ratio is None:
        return None

    try:
        ratio = float(disorder_ratio)
    except (ValueError, TypeError):
        return None

    if ratio < 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_presortedness_detection_before_compare",
            severity=min(1.0, (0.1 - ratio) / 0.1),
            evidence=(
                f"Disorder ratio {ratio:.2f} is below 0.1 threshold "
                f"— input is nearly sorted, adaptive algorithm recommended"
            ),
            metric_name="disorder_ratio",
            metric_value=ratio,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_comparison_count(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect excessive comparison count."""
    intermediates = context.intermediates or {}
    n_comparisons = intermediates.get("n_comparisons")
    n_elements = intermediates.get("n_elements")

    if n_comparisons is None or n_elements is None:
        return None

    try:
        comps = int(n_comparisons)
        n = int(n_elements)
    except (ValueError, TypeError):
        return None

    if n <= 1:
        return None

    expected = 2.0 * n * np.log2(max(n, 2))
    ratio = comps / expected

    if ratio > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_comparison_count_analysis_after_compare",
            severity=min(1.0, (ratio - 1.0) / 2.0),
            evidence=(
                f"Comparison ratio {ratio:.2f} exceeds 1.0 "
                f"({comps} comparisons vs {expected:.0f} expected for n={n})"
            ),
            metric_name="comparison_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_swap_count(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect excessive swap/move count."""
    intermediates = context.intermediates or {}
    n_swaps = intermediates.get("n_swaps")
    n_elements = intermediates.get("n_elements")

    if n_swaps is None or n_elements is None:
        return None

    try:
        swaps = int(n_swaps)
        n = int(n_elements)
    except (ValueError, TypeError):
        return None

    if n <= 1:
        return None

    expected = 2.0 * n * np.log2(max(n, 2))
    ratio = swaps / expected

    if ratio > 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_swap_count_analysis_after_swap",
            severity=min(1.0, (ratio - 1.0) / 2.0),
            evidence=(
                f"Swap ratio {ratio:.2f} exceeds 1.0 "
                f"({swaps} swaps vs {expected:.0f} expected for n={n})"
            ),
            metric_name="swap_ratio",
            metric_value=ratio,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_stability(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect stability violations in sort output."""
    intermediates = context.intermediates or {}
    n_stability_violations = intermediates.get("n_stability_violations")

    if n_stability_violations is None:
        return None

    try:
        violations = int(n_stability_violations)
    except (ValueError, TypeError):
        return None

    if violations > 0:
        return ExpansionDiagnostic(
            rule_name="insert_stability_validation_after_recurse",
            severity=min(1.0, violations / 10.0),
            evidence=(
                f"{violations} stability violation(s) detected "
                f"— equal-key relative order not preserved"
            ),
            metric_name="n_stability_violations",
            metric_value=float(violations),
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SortingExpansionRuleSet:
    """Expansion rules for sorting pipelines (Insertion, Heapsort, Quicksort, Merge Sort)."""

    name = "sorting"
    domain = "sorting"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_presortedness_detection(),
            _build_insert_comparison_count_analysis(),
            _build_insert_swap_count_analysis(),
            _build_insert_stability_validation(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        presort = _diagnose_presortedness(cdg, context)
        if presort is not None:
            diagnostics.append(presort)

        comps = _diagnose_comparison_count(cdg, context)
        if comps is not None:
            diagnostics.append(comps)

        swaps = _diagnose_swap_count(cdg, context)
        if swaps is not None:
            diagnostics.append(swaps)

        stability = _diagnose_stability(cdg, context)
        if stability is not None:
            diagnostics.append(stability)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

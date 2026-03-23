"""Expansion rules for the Greedy family (Huffman, Activity Selector, Knapsack, Prim).

Defines DPO rules and diagnostic functions that let the expansion engine
insert matroid validation, tie detection, solution quality bounds, and
redundant feasibility detection into greedy CDGs.

Greedy skeleton topology (4 nodes, linear pipeline):

    Sort Candidates → Greedy Choice → Feasibility Check → Update Solution

Expansion insertion points:
  - Before Greedy Choice: matroid exchange validation
  - After Sort Candidates: criterion tie detection
  - After Update Solution: solution quality bound
  - Before Feasibility Check: redundant feasibility detection

All diagnostics are pure functions of greedy intermediates.
"""

from __future__ import annotations

import logging

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

_DOMAIN = "greedy"

# Greedy skeleton node names
_SORT_CANDIDATES = "Sort Candidates"
_GREEDY_CHOICE = "Greedy Choice"
_FEASIBILITY_CHECK = "Feasibility Check"
_UPDATE_SOLUTION = "Update Solution"


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


def _build_insert_matroid_validation() -> RewriteRule:
    """Interpose ``validate_matroid_exchange`` before Greedy Choice.

    For non-matroid problems, the greedy strategy may produce suboptimal
    results.  This pre-check validates the exchange property so the user
    is warned when greedy optimality is not guaranteed.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    choice = _node(
        "choice",
        _GREEDY_CHOICE,
        ConceptType.GREEDY,
    )
    lhs = CDGExport(nodes=[src, choice], edges=[_edge("src", "choice")])
    interface = CDGExport(nodes=[src, choice], edges=[])

    matroid = _node(
        "matroid",
        "Validate Matroid Exchange",
        ConceptType.GREEDY,
        matched_primitive="validate_matroid_exchange",
        inputs=[
            IOSpec(name="selected_sets", type_desc="list[ndarray]"),
            IOSpec(name="ground_set_size", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="exchange_ratio", type_desc="float"),
            IOSpec(name="is_matroid", type_desc="bool"),
        ],
        description="Check the exchange property on observed selection sets.",
        type_signature="list[ndarray], int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, matroid, choice],
        edges=[
            _edge("src", "matroid"),
            _edge("matroid", "choice"),
        ],
    )

    return RewriteRule(
        name="insert_matroid_validation_before_greedy_choice",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "choice": "choice"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "choice": "choice"}, edge_map={}),
        priority=3,
    )


def _build_insert_tie_detection() -> RewriteRule:
    """Interpose ``detect_criterion_ties`` after Sort Candidates.

    Near-ties in the greedy criterion make the selection unstable —
    different tie-breaking strategies can produce different solutions.
    """
    sort = _node(
        "sort",
        _SORT_CANDIDATES,
        ConceptType.GREEDY,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[sort, sink], edges=[_edge("sort", "sink")])
    interface = CDGExport(nodes=[sort, sink], edges=[])

    ties = _node(
        "ties",
        "Detect Criterion Ties",
        ConceptType.GREEDY,
        matched_primitive="detect_criterion_ties",
        inputs=[
            IOSpec(name="scores", type_desc="ndarray"),
            IOSpec(name="tie_tolerance", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="n_ties", type_desc="int"),
            IOSpec(name="tie_groups", type_desc="ndarray"),
        ],
        description="Detect near-ties in greedy criterion ordering.",
        type_signature="ndarray, float -> tuple[int, ndarray]",
    )
    rhs = CDGExport(
        nodes=[sort, ties, sink],
        edges=[
            _edge("sort", "ties"),
            _edge("ties", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_tie_detection_after_sort",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"sort": "sort", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"sort": "sort", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_quality_bound() -> RewriteRule:
    """Interpose ``estimate_solution_quality`` after Update Solution.

    When the greedy solution is significantly worse than the relaxation
    bound, the user should consider alternative algorithms.
    """
    update = _node(
        "update",
        _UPDATE_SOLUTION,
        ConceptType.GREEDY,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    quality = _node(
        "quality",
        "Estimate Solution Quality",
        ConceptType.GREEDY,
        matched_primitive="estimate_solution_quality",
        inputs=[
            IOSpec(name="greedy_value", type_desc="float"),
            IOSpec(name="relaxation_bound", type_desc="float"),
        ],
        outputs=[
            IOSpec(name="approx_ratio", type_desc="float"),
            IOSpec(name="is_optimal", type_desc="bool"),
        ],
        description="Compute approximation ratio of greedy solution against a known bound.",
        type_signature="float, float -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[update, quality, sink],
        edges=[
            _edge("update", "quality"),
            _edge("quality", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_quality_bound_after_update",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _build_insert_redundant_feasibility_detection() -> RewriteRule:
    """Interpose ``detect_redundant_feasibility`` before Feasibility Check.

    When all feasibility checks pass (monotone constraints), the check
    can potentially be skipped for performance.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    feasibility = _node(
        "feasibility",
        _FEASIBILITY_CHECK,
        ConceptType.GREEDY,
    )
    lhs = CDGExport(nodes=[src, feasibility], edges=[_edge("src", "feasibility")])
    interface = CDGExport(nodes=[src, feasibility], edges=[])

    redundant = _node(
        "redundant",
        "Detect Redundant Feasibility",
        ConceptType.GREEDY,
        matched_primitive="detect_redundant_feasibility",
        inputs=[
            IOSpec(name="feasibility_history", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="pass_rate", type_desc="float"),
            IOSpec(name="is_redundant", type_desc="bool"),
        ],
        description="Detect when feasibility checks are always passing.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[src, redundant, feasibility],
        edges=[
            _edge("src", "redundant"),
            _edge("redundant", "feasibility"),
        ],
    )

    return RewriteRule(
        name="insert_redundant_feasibility_detection_before_feasibility",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "feasibility": "feasibility"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "feasibility": "feasibility"}, edge_map={}),
        priority=2,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_matroid_violation(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect matroid exchange property violations."""
    intermediates = context.intermediates or {}
    exchange_ratio = intermediates.get("exchange_ratio")

    if exchange_ratio is None:
        return None

    try:
        ratio = float(exchange_ratio)
    except (ValueError, TypeError):
        return None

    if ratio < 0.95:
        return ExpansionDiagnostic(
            rule_name="insert_matroid_validation_before_greedy_choice",
            severity=min(1.0, (0.95 - ratio) / 0.95),
            evidence=(
                f"Exchange ratio {ratio:.2f} is below 0.95 threshold "
                f"— greedy may not produce optimal results"
            ),
            metric_name="exchange_ratio",
            metric_value=ratio,
            threshold=0.95,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_criterion_ties(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect high fraction of tied candidates in greedy criterion."""
    intermediates = context.intermediates or {}
    tie_fraction = intermediates.get("tie_fraction")

    if tie_fraction is None:
        return None

    try:
        fraction = float(tie_fraction)
    except (ValueError, TypeError):
        return None

    if fraction > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_tie_detection_after_sort",
            severity=min(1.0, (fraction - 0.1) / 0.9),
            evidence=(
                f"Tie fraction {fraction:.2f} exceeds 0.1 threshold "
                f"— {fraction * 100:.0f}% of candidates have near-equal scores"
            ),
            metric_name="tie_fraction",
            metric_value=fraction,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_solution_quality(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect poor greedy solution quality relative to relaxation bound."""
    intermediates = context.intermediates or {}
    greedy_value = intermediates.get("greedy_value")
    relaxation_bound = intermediates.get("relaxation_bound")

    if greedy_value is None or relaxation_bound is None:
        return None

    try:
        gv = float(greedy_value)
        rb = float(relaxation_bound)
    except (ValueError, TypeError):
        return None

    if rb == 0.0:
        return None

    approx_ratio = gv / rb
    approx_ratio = max(0.0, min(1.0, approx_ratio))

    if approx_ratio < 0.9:
        return ExpansionDiagnostic(
            rule_name="insert_quality_bound_after_update",
            severity=min(1.0, (0.9 - approx_ratio) / 0.9),
            evidence=(
                f"Approximation ratio {approx_ratio:.2f} is below 0.9 threshold "
                f"— greedy solution is worse than 90% of relaxation bound"
            ),
            metric_name="approx_ratio",
            metric_value=approx_ratio,
            threshold=0.9,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_redundant_feasibility(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect when all feasibility checks are passing."""
    intermediates = context.intermediates or {}
    feasibility_pass_rate = intermediates.get("feasibility_pass_rate")

    if feasibility_pass_rate is None:
        return None

    try:
        rate = float(feasibility_pass_rate)
    except (ValueError, TypeError):
        return None

    if rate == 1.0:
        return ExpansionDiagnostic(
            rule_name="insert_redundant_feasibility_detection_before_feasibility",
            severity=0.5,
            evidence=(
                "Feasibility pass rate is 1.0 — all checks passing, "
                "feasibility check may be redundant"
            ),
            metric_name="feasibility_pass_rate",
            metric_value=rate,
            threshold=1.0,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class GreedyExpansionRuleSet:
    """Expansion rules for greedy pipelines (Huffman, Activity Selector, Knapsack, Prim)."""

    name = "greedy"
    domain = "greedy"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_matroid_validation(),
            _build_insert_tie_detection(),
            _build_insert_quality_bound(),
            _build_insert_redundant_feasibility_detection(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        matroid = _diagnose_matroid_violation(cdg, context)
        if matroid is not None:
            diagnostics.append(matroid)

        ties = _diagnose_criterion_ties(cdg, context)
        if ties is not None:
            diagnostics.append(ties)

        quality = _diagnose_solution_quality(cdg, context)
        if quality is not None:
            diagnostics.append(quality)

        redundant = _diagnose_redundant_feasibility(cdg, context)
        if redundant is not None:
            diagnostics.append(redundant)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

"""Expansion rules for the Graph Signal Processing family.

Graph Signal Processing skeleton topology (4 nodes, linear pipeline):

    Build Graph → Compute Laplacian → GFT → Graph Filter/Diffuse

Expansion insertion points:
  - After Build Graph: connectivity validation
  - After Compute Laplacian: symmetry check
  - After GFT: spectral gap analysis
  - After Graph Filter/Diffuse: filter response validation
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

_DOMAIN = "graph_signal_processing"

_BUILD_GRAPH = "Build Graph"
_COMPUTE_LAPLACIAN = "Compute Laplacian"
_GFT = "GFT"
_GRAPH_FILTER = "Graph Filter/Diffuse"


def _node(
    node_id: str, name: str, concept_type: ConceptType, *,
    matched_primitive: str | None = None, inputs: list[IOSpec] | None = None,
    outputs: list[IOSpec] | None = None, description: str = "",
    type_signature: str = "",
) -> AlgorithmicNode:
    return AlgorithmicNode(
        node_id=node_id, name=name, description=description or name,
        concept_type=concept_type, status=NodeStatus.ATOMIC,
        matched_primitive=matched_primitive, inputs=inputs or [],
        outputs=outputs or [], type_signature=type_signature or f"{name} -> result",
    )


def _edge(
    source_id: str, target_id: str, output_name: str = "out",
    input_name: str = "in", type_desc: str = "ndarray",
) -> DependencyEdge:
    return DependencyEdge(
        source_id=source_id, target_id=target_id, output_name=output_name,
        input_name=input_name, source_type=type_desc, target_type=type_desc,
    )


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_connectivity_validation() -> RewriteRule:
    build = _node("build", _BUILD_GRAPH, ConceptType.GRAPH_SIGNAL_PROCESSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[build, sink], edges=[_edge("build", "sink")])
    interface = CDGExport(nodes=[build, sink], edges=[])

    connectivity = _node(
        "connectivity", "Validate Graph Connectivity", ConceptType.GRAPH_SIGNAL_PROCESSING,
        matched_primitive="validate_graph_connectivity",
        inputs=[IOSpec(name="adjacency", type_desc="ndarray")],
        outputs=[IOSpec(name="n_components", type_desc="int"), IOSpec(name="is_connected", type_desc="bool")],
        description="Validate that the graph is connected.",
        type_signature="ndarray -> tuple[int, bool]",
    )
    rhs = CDGExport(nodes=[build, connectivity, sink], edges=[_edge("build", "connectivity"), _edge("connectivity", "sink")])

    return RewriteRule(
        name="insert_connectivity_validation_after_build_graph", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"build": "build", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"build": "build", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_laplacian_symmetry_check() -> RewriteRule:
    laplacian = _node("laplacian", _COMPUTE_LAPLACIAN, ConceptType.GRAPH_SIGNAL_PROCESSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[laplacian, sink], edges=[_edge("laplacian", "sink")])
    interface = CDGExport(nodes=[laplacian, sink], edges=[])

    symmetry = _node(
        "symmetry", "Check Laplacian Symmetry", ConceptType.GRAPH_SIGNAL_PROCESSING,
        matched_primitive="check_laplacian_symmetry",
        inputs=[IOSpec(name="laplacian", type_desc="ndarray"), IOSpec(name="tolerance", type_desc="float")],
        outputs=[IOSpec(name="max_asymmetry", type_desc="float"), IOSpec(name="is_symmetric", type_desc="bool")],
        description="Check that the graph Laplacian is symmetric.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[laplacian, symmetry, sink], edges=[_edge("laplacian", "symmetry"), _edge("symmetry", "sink")])

    return RewriteRule(
        name="insert_laplacian_symmetry_check_after_compute_laplacian", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"laplacian": "laplacian", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"laplacian": "laplacian", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_spectral_gap_analysis() -> RewriteRule:
    gft = _node("gft", _GFT, ConceptType.GRAPH_SIGNAL_PROCESSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[gft, sink], edges=[_edge("gft", "sink")])
    interface = CDGExport(nodes=[gft, sink], edges=[])

    gap = _node(
        "gap", "Analyze Spectral Gap", ConceptType.GRAPH_SIGNAL_PROCESSING,
        matched_primitive="analyze_spectral_gap",
        inputs=[IOSpec(name="eigenvalues", type_desc="ndarray")],
        outputs=[IOSpec(name="spectral_gap", type_desc="float"), IOSpec(name="is_well_connected", type_desc="bool")],
        description="Analyze the spectral gap (algebraic connectivity) of the graph.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[gft, gap, sink], edges=[_edge("gft", "gap"), _edge("gap", "sink")])

    return RewriteRule(
        name="insert_spectral_gap_analysis_after_gft", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"gft": "gft", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"gft": "gft", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_filter_response_validation() -> RewriteRule:
    gfilter = _node("gfilter", _GRAPH_FILTER, ConceptType.GRAPH_SIGNAL_PROCESSING)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[gfilter, sink], edges=[_edge("gfilter", "sink")])
    interface = CDGExport(nodes=[gfilter, sink], edges=[])

    validate = _node(
        "validate", "Validate Filter Response", ConceptType.GRAPH_SIGNAL_PROCESSING,
        matched_primitive="validate_filter_response",
        inputs=[IOSpec(name="filter_response", type_desc="ndarray"), IOSpec(name="eigenvalues", type_desc="ndarray")],
        outputs=[IOSpec(name="max_gain", type_desc="float"), IOSpec(name="is_stable", type_desc="bool")],
        description="Validate that the graph filter response is well-behaved.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[gfilter, validate, sink], edges=[_edge("gfilter", "validate"), _edge("validate", "sink")])

    return RewriteRule(
        name="insert_filter_response_validation_after_graph_filter", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"gfilter": "gfilter", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"gfilter": "gfilter", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_connectivity(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    n_comp = intermediates.get("n_graph_components")
    if n_comp is None:
        return None
    try:
        c = int(n_comp)
    except (ValueError, TypeError):
        return None
    if c > 1:
        return ExpansionDiagnostic(
            rule_name="insert_connectivity_validation_after_build_graph",
            severity=min(1.0, c / 5.0), evidence=f"Graph has {c} components — not connected",
            metric_name="n_graph_components", metric_value=float(c), threshold=1.0, source_domain=_DOMAIN,
        )
    return None


def _diagnose_laplacian_symmetry(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    asym = intermediates.get("laplacian_max_asymmetry")
    if asym is None:
        return None
    try:
        a = float(asym)
    except (ValueError, TypeError):
        return None
    if a > 1e-10:
        return ExpansionDiagnostic(
            rule_name="insert_laplacian_symmetry_check_after_compute_laplacian",
            severity=min(1.0, a * 1e8), evidence=f"Laplacian asymmetry {a:.2e} exceeds 1e-10",
            metric_name="laplacian_max_asymmetry", metric_value=a, threshold=1e-10, source_domain=_DOMAIN,
        )
    return None


def _diagnose_spectral_gap(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    gap = intermediates.get("spectral_gap")
    if gap is None:
        return None
    try:
        g = float(gap)
    except (ValueError, TypeError):
        return None
    if g < 0.01:
        return ExpansionDiagnostic(
            rule_name="insert_spectral_gap_analysis_after_gft",
            severity=min(1.0, (0.01 - g) / 0.01), evidence=f"Spectral gap {g:.4f} below 0.01 — near-disconnected graph",
            metric_name="spectral_gap", metric_value=g, threshold=0.01, source_domain=_DOMAIN,
        )
    return None


def _diagnose_filter_response(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    gain = intermediates.get("max_filter_gain")
    if gain is None:
        return None
    try:
        g = float(gain)
    except (ValueError, TypeError):
        return None
    if g > 100.0:
        return ExpansionDiagnostic(
            rule_name="insert_filter_response_validation_after_graph_filter",
            severity=min(1.0, g / 1000.0), evidence=f"Max filter gain {g:.1f} exceeds 100 — unstable filter",
            metric_name="max_filter_gain", metric_value=g, threshold=100.0, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class GraphSignalProcessingExpansionRuleSet:
    """Expansion rules for graph signal processing pipelines."""

    name = "graph_signal_processing"
    domain = "graph_signal_processing"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_connectivity_validation(),
            _build_insert_laplacian_symmetry_check(),
            _build_insert_spectral_gap_analysis(),
            _build_insert_filter_response_validation(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_connectivity, _diagnose_laplacian_symmetry, _diagnose_spectral_gap, _diagnose_filter_response]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

"""Expansion rules for the Signal Transform family (FFT Filter, Spectral Analysis, DCT, STFT).

Signal Transform skeleton topology (4 nodes, linear pipeline):

    Window → Forward Transform → Spectral Processing → Inverse Transform

Expansion insertion points:
  - After Window: window leakage analysis
  - After Forward Transform: spectral aliasing detection
  - After Spectral Processing: Parseval energy validation
  - After Inverse Transform: inverse reconstruction check
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

_DOMAIN = "signal_transform"

_WINDOW = "Window"
_FORWARD_TRANSFORM = "Forward Transform"
_SPECTRAL_PROCESSING = "Spectral Processing"
_INVERSE_TRANSFORM = "Inverse Transform"


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


def _build_insert_window_leakage_analysis() -> RewriteRule:
    window = _node("window", _WINDOW, ConceptType.SIGNAL_TRANSFORM)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[window, sink], edges=[_edge("window", "sink")])
    interface = CDGExport(nodes=[window, sink], edges=[])

    leakage = _node(
        "leakage", "Analyze Window Leakage", ConceptType.SIGNAL_TRANSFORM,
        matched_primitive="analyze_window_leakage",
        inputs=[IOSpec(name="windowed", type_desc="ndarray"), IOSpec(name="original", type_desc="ndarray")],
        outputs=[IOSpec(name="leakage_ratio", type_desc="float"), IOSpec(name="is_excessive", type_desc="bool")],
        description="Analyze spectral leakage introduced by the window function.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[window, leakage, sink], edges=[_edge("window", "leakage"), _edge("leakage", "sink")])

    return RewriteRule(
        name="insert_window_leakage_analysis_after_window", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"window": "window", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"window": "window", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_aliasing_detection() -> RewriteRule:
    forward = _node("forward", _FORWARD_TRANSFORM, ConceptType.SIGNAL_TRANSFORM)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[forward, sink], edges=[_edge("forward", "sink")])
    interface = CDGExport(nodes=[forward, sink], edges=[])

    aliasing = _node(
        "aliasing", "Detect Spectral Aliasing", ConceptType.SIGNAL_TRANSFORM,
        matched_primitive="detect_spectral_aliasing",
        inputs=[IOSpec(name="spectrum", type_desc="ndarray"), IOSpec(name="nyquist_fraction", type_desc="float")],
        outputs=[IOSpec(name="alias_energy_fraction", type_desc="float"), IOSpec(name="has_aliasing", type_desc="bool")],
        description="Detect potential aliasing by checking energy near Nyquist.",
        type_signature="ndarray, float -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[forward, aliasing, sink], edges=[_edge("forward", "aliasing"), _edge("aliasing", "sink")])

    return RewriteRule(
        name="insert_aliasing_detection_after_forward_transform", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"forward": "forward", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"forward": "forward", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_parseval_validation() -> RewriteRule:
    spectral = _node("spectral", _SPECTRAL_PROCESSING, ConceptType.SIGNAL_TRANSFORM)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[spectral, sink], edges=[_edge("spectral", "sink")])
    interface = CDGExport(nodes=[spectral, sink], edges=[])

    parseval = _node(
        "parseval", "Validate Parseval Energy", ConceptType.SIGNAL_TRANSFORM,
        matched_primitive="validate_parseval_energy",
        inputs=[IOSpec(name="time_domain", type_desc="ndarray"), IOSpec(name="freq_domain", type_desc="ndarray")],
        outputs=[IOSpec(name="relative_error", type_desc="float"), IOSpec(name="is_valid", type_desc="bool")],
        description="Validate energy conservation between time and frequency domains.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[spectral, parseval, sink], edges=[_edge("spectral", "parseval"), _edge("parseval", "sink")])

    return RewriteRule(
        name="insert_parseval_validation_after_spectral_processing", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"spectral": "spectral", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"spectral": "spectral", "sink": "sink"}, edge_map={}),
        priority=1,
    )


def _build_insert_reconstruction_check() -> RewriteRule:
    inverse = _node("inverse", _INVERSE_TRANSFORM, ConceptType.SIGNAL_TRANSFORM)
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[inverse, sink], edges=[_edge("inverse", "sink")])
    interface = CDGExport(nodes=[inverse, sink], edges=[])

    recon = _node(
        "recon", "Check Inverse Reconstruction", ConceptType.SIGNAL_TRANSFORM,
        matched_primitive="check_inverse_reconstruction",
        inputs=[IOSpec(name="original", type_desc="ndarray"), IOSpec(name="reconstructed", type_desc="ndarray")],
        outputs=[IOSpec(name="relative_error", type_desc="float"), IOSpec(name="is_faithful", type_desc="bool")],
        description="Check round-trip reconstruction quality of forward+inverse transform.",
        type_signature="ndarray, ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(nodes=[inverse, recon, sink], edges=[_edge("inverse", "recon"), _edge("recon", "sink")])

    return RewriteRule(
        name="insert_reconstruction_check_after_inverse", lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"inverse": "inverse", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"inverse": "inverse", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_window_leakage(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    ratio = intermediates.get("window_leakage_ratio")
    if ratio is None:
        return None
    try:
        r = float(ratio)
    except (ValueError, TypeError):
        return None
    if r > 0.5:
        return ExpansionDiagnostic(
            rule_name="insert_window_leakage_analysis_after_window",
            severity=min(1.0, r), evidence=f"Window leakage ratio {r:.2f} exceeds 0.5 — excessive attenuation",
            metric_name="window_leakage_ratio", metric_value=r, threshold=0.5, source_domain=_DOMAIN,
        )
    return None


def _diagnose_aliasing(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    frac = intermediates.get("alias_energy_fraction")
    if frac is None:
        return None
    try:
        f = float(frac)
    except (ValueError, TypeError):
        return None
    if f > 0.1:
        return ExpansionDiagnostic(
            rule_name="insert_aliasing_detection_after_forward_transform",
            severity=min(1.0, f * 5), evidence=f"Alias energy fraction {f:.3f} exceeds 0.1 — potential aliasing",
            metric_name="alias_energy_fraction", metric_value=f, threshold=0.1, source_domain=_DOMAIN,
        )
    return None


def _diagnose_parseval(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    err = intermediates.get("parseval_relative_error")
    if err is None:
        return None
    try:
        e = float(err)
    except (ValueError, TypeError):
        return None
    if e > 1e-6:
        return ExpansionDiagnostic(
            rule_name="insert_parseval_validation_after_spectral_processing",
            severity=min(1.0, e * 1e4), evidence=f"Parseval relative error {e:.2e} exceeds 1e-6 — energy not conserved",
            metric_name="parseval_relative_error", metric_value=e, threshold=1e-6, source_domain=_DOMAIN,
        )
    return None


def _diagnose_reconstruction(cdg: CDGExport, context: ExpansionContext) -> ExpansionDiagnostic | None:
    intermediates = context.intermediates or {}
    err = intermediates.get("reconstruction_error")
    if err is None:
        return None
    try:
        e = float(err)
    except (ValueError, TypeError):
        return None
    if e > 1e-10:
        return ExpansionDiagnostic(
            rule_name="insert_reconstruction_check_after_inverse",
            severity=min(1.0, e * 1e8), evidence=f"Reconstruction error {e:.2e} exceeds 1e-10 — lossy round-trip",
            metric_name="reconstruction_error", metric_value=e, threshold=1e-10, source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SignalTransformExpansionRuleSet:
    """Expansion rules for signal transform pipelines (FFT, DCT, STFT)."""

    name = "signal_transform"
    domain = "signal_transform"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_window_leakage_analysis(),
            _build_insert_aliasing_detection(),
            _build_insert_parseval_validation(),
            _build_insert_reconstruction_check(),
        ]

    def diagnose(self, cdg: CDGExport, context: ExpansionContext) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [_diagnose_window_leakage, _diagnose_aliasing, _diagnose_parseval, _diagnose_reconstruction]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

"""Expansion rules for the Baseline Analysis family.

Baseline analysis skeleton topology (8 nodes, linear pipeline):

    Acquire Data -> Preprocess -> Windowed Analysis -> Fit ->
    Output Transform -> Normalize -> Combine -> Regionize

Expansion insertion points:
  - After Fit: onset coverage check
  - After Output Transform: padding saturation detection
  - After Normalize: normalization clipping monitoring
  - After Combine: component balance validation
"""

from __future__ import annotations

from sciona.architect.graph_rewriter import Morphism, RewriteRule
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionDiagnostic

_DOMAIN = "baseline_analysis"

_FIT = "Fit"
_OUTPUT_TRANSFORM = "Output Transform"
_NORMALIZE = "Normalize"
_COMBINE = "Combine"
_REGIONIZE = "Regionize"

_FIT_EDGE = "fit->transform"
_TRANSFORM_EDGE = "transform->normalize"
_NORMALIZE_EDGE = "normalize->combine"


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


def _fit_node(node_id: str = "fit") -> AlgorithmicNode:
    return _node(
        node_id,
        _FIT,
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="baseline_fit_stack",
    )


def _baseline_node(node_id: str, name: str) -> AlgorithmicNode:
    return _node(node_id, name, ConceptType.BASELINE_ANALYSIS)


def _build_insert_onset_coverage_check() -> RewriteRule:
    fit = _fit_node()
    sink = _baseline_node("sink", _OUTPUT_TRANSFORM)
    lhs = CDGExport(nodes=[fit, sink], edges=[_edge("fit", "sink")])
    interface = CDGExport(nodes=[fit, sink], edges=[])

    onset = _node(
        "onset",
        "Check Onset Coverage",
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="check_onset_coverage",
        inputs=[
            IOSpec(name="fit_results", type_desc="list"),
            IOSpec(name="signal_length", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="onset_density", type_desc="float"),
            IOSpec(name="has_sufficient_onsets", type_desc="bool"),
        ],
        description="Check onset detection density relative to signal length.",
        type_signature="list, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[fit, onset, sink],
        edges=[_edge("fit", "onset"), _edge("onset", "sink")],
    )

    return RewriteRule(
        name="insert_onset_coverage_check_after_fit",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"fit": "fit", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"fit": "fit", "sink": "sink"}, edge_map={}),
        priority=3,
    )


def _build_insert_padding_saturation() -> RewriteRule:
    fit = _fit_node()
    transform = _baseline_node("transform", _OUTPUT_TRANSFORM)
    normalize = _baseline_node("normalize", _NORMALIZE)
    lhs = CDGExport(
        nodes=[fit, transform, normalize],
        edges=[_edge("fit", "transform"), _edge("transform", "normalize")],
    )
    interface = CDGExport(
        nodes=[fit, transform, normalize],
        edges=[_edge("fit", "transform")],
    )

    padding = _node(
        "padding",
        "Detect Padding Saturation",
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="detect_padding_saturation",
        inputs=[
            IOSpec(name="padded", type_desc="ndarray"),
            IOSpec(name="original_length", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="padding_overlap_fraction", type_desc="float"),
            IOSpec(name="is_saturated", type_desc="bool"),
        ],
        description="Detect excessive padding fraction in output signal.",
        type_signature="ndarray, int -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[fit, transform, padding, normalize],
        edges=[
            _edge("fit", "transform"),
            _edge("transform", "padding"),
            _edge("padding", "normalize"),
        ],
    )

    return RewriteRule(
        name="insert_padding_saturation_after_transform",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={
                "fit": "fit",
                "transform": "transform",
                "normalize": "normalize",
            },
            edge_map={_FIT_EDGE: _FIT_EDGE},
        ),
        r_morphism=Morphism(
            node_map={
                "fit": "fit",
                "transform": "transform",
                "normalize": "normalize",
            },
            edge_map={_FIT_EDGE: _FIT_EDGE},
        ),
        priority=2,
    )


def _build_insert_normalization_clipping() -> RewriteRule:
    fit = _fit_node()
    transform = _baseline_node("transform", _OUTPUT_TRANSFORM)
    normalize = _baseline_node("normalize", _NORMALIZE)
    combine = _baseline_node("combine", _COMBINE)
    lhs = CDGExport(
        nodes=[fit, transform, normalize, combine],
        edges=[
            _edge("fit", "transform"),
            _edge("transform", "normalize"),
            _edge("normalize", "combine"),
        ],
    )
    interface = CDGExport(
        nodes=[fit, transform, normalize, combine],
        edges=[_edge("fit", "transform"), _edge("transform", "normalize")],
    )

    clipping = _node(
        "clipping",
        "Monitor Normalization Clipping",
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="monitor_normalization_clipping",
        inputs=[IOSpec(name="normalized", type_desc="ndarray")],
        outputs=[
            IOSpec(name="clipped_fraction", type_desc="float"),
            IOSpec(name="is_clipped", type_desc="bool"),
        ],
        description="Monitor fraction of normalized values clipped at ceiling.",
        type_signature="ndarray -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[fit, transform, normalize, clipping, combine],
        edges=[
            _edge("fit", "transform"),
            _edge("transform", "normalize"),
            _edge("normalize", "clipping"),
            _edge("clipping", "combine"),
        ],
    )

    return RewriteRule(
        name="insert_normalization_clipping_after_normalize",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={
                "fit": "fit",
                "transform": "transform",
                "normalize": "normalize",
                "combine": "combine",
            },
            edge_map={
                _FIT_EDGE: _FIT_EDGE,
                _TRANSFORM_EDGE: _TRANSFORM_EDGE,
            },
        ),
        r_morphism=Morphism(
            node_map={
                "fit": "fit",
                "transform": "transform",
                "normalize": "normalize",
                "combine": "combine",
            },
            edge_map={
                _FIT_EDGE: _FIT_EDGE,
                _TRANSFORM_EDGE: _TRANSFORM_EDGE,
            },
        ),
        priority=2,
    )


def _build_insert_component_balance() -> RewriteRule:
    fit = _fit_node()
    transform = _baseline_node("transform", _OUTPUT_TRANSFORM)
    normalize = _baseline_node("normalize", _NORMALIZE)
    combine = _baseline_node("combine", _COMBINE)
    regionize = _baseline_node("regionize", _REGIONIZE)
    lhs = CDGExport(
        nodes=[fit, transform, normalize, combine, regionize],
        edges=[
            _edge("fit", "transform"),
            _edge("transform", "normalize"),
            _edge("normalize", "combine"),
            _edge("combine", "regionize"),
        ],
    )
    interface = CDGExport(
        nodes=[fit, transform, normalize, combine, regionize],
        edges=[
            _edge("fit", "transform"),
            _edge("transform", "normalize"),
            _edge("normalize", "combine"),
        ],
    )

    balance = _node(
        "balance",
        "Validate Component Balance",
        ConceptType.BASELINE_ANALYSIS,
        matched_primitive="validate_component_balance",
        inputs=[IOSpec(name="component_outputs", type_desc="list[ndarray]")],
        outputs=[
            IOSpec(name="component_entropy", type_desc="float"),
            IOSpec(name="is_balanced", type_desc="bool"),
        ],
        description="Validate energy balance across component contributions.",
        type_signature="list[ndarray] -> tuple[float, bool]",
    )
    rhs = CDGExport(
        nodes=[fit, transform, normalize, combine, balance, regionize],
        edges=[
            _edge("fit", "transform"),
            _edge("transform", "normalize"),
            _edge("normalize", "combine"),
            _edge("combine", "balance"),
            _edge("balance", "regionize"),
        ],
    )

    return RewriteRule(
        name="insert_component_balance_after_combine",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={
                "fit": "fit",
                "transform": "transform",
                "normalize": "normalize",
                "combine": "combine",
                "regionize": "regionize",
            },
            edge_map={
                _FIT_EDGE: _FIT_EDGE,
                _TRANSFORM_EDGE: _TRANSFORM_EDGE,
                _NORMALIZE_EDGE: _NORMALIZE_EDGE,
            },
        ),
        r_morphism=Morphism(
            node_map={
                "fit": "fit",
                "transform": "transform",
                "normalize": "normalize",
                "combine": "combine",
                "regionize": "regionize",
            },
            edge_map={
                _FIT_EDGE: _FIT_EDGE,
                _TRANSFORM_EDGE: _TRANSFORM_EDGE,
                _NORMALIZE_EDGE: _NORMALIZE_EDGE,
            },
        ),
        priority=1,
    )


def _diagnose_onset_coverage(
    cdg: CDGExport,
    context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    del cdg
    value = (context.intermediates or {}).get("onset_density")
    if value is None:
        return None
    try:
        density = float(value)
    except (TypeError, ValueError):
        return None
    if density < 1e-4:
        severity = min(1.0, max(0.0, (1e-4 - density) / 1e-4))
        return ExpansionDiagnostic(
            rule_name="insert_onset_coverage_check_after_fit",
            severity=severity,
            evidence=f"Onset density {density:.2e} below 1e-4 threshold",
            metric_name="onset_density",
            metric_value=density,
            threshold=1e-4,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_padding_saturation(
    cdg: CDGExport,
    context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    del cdg
    value = (context.intermediates or {}).get("padding_overlap_fraction")
    if value is None:
        return None
    try:
        fraction = float(value)
    except (TypeError, ValueError):
        return None
    if fraction > 0.5:
        severity = min(1.0, max(0.0, (fraction - 0.5) / 0.5))
        return ExpansionDiagnostic(
            rule_name="insert_padding_saturation_after_transform",
            severity=severity,
            evidence=f"Padding fraction {fraction:.2%} exceeds 50% threshold",
            metric_name="padding_overlap_fraction",
            metric_value=fraction,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_normalization_clipping(
    cdg: CDGExport,
    context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    del cdg
    value = (context.intermediates or {}).get("clipped_fraction")
    if value is None:
        return None
    try:
        clipped = float(value)
    except (TypeError, ValueError):
        return None
    if clipped > 0.1:
        severity = min(1.0, max(0.0, (clipped - 0.1) / 0.9))
        return ExpansionDiagnostic(
            rule_name="insert_normalization_clipping_after_normalize",
            severity=severity,
            evidence=f"Clipped fraction {clipped:.2%} exceeds 10% threshold",
            metric_name="clipped_fraction",
            metric_value=clipped,
            threshold=0.1,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_component_balance(
    cdg: CDGExport,
    context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    del cdg
    value = (context.intermediates or {}).get("component_entropy")
    if value is None:
        return None
    try:
        entropy = float(value)
    except (TypeError, ValueError):
        return None
    if entropy < 0.5:
        severity = min(1.0, max(0.0, (0.5 - entropy) / 0.5))
        return ExpansionDiagnostic(
            rule_name="insert_component_balance_after_combine",
            severity=severity,
            evidence=f"Component entropy {entropy:.3f} below 0.5 threshold",
            metric_name="component_entropy",
            metric_value=entropy,
            threshold=0.5,
            source_domain=_DOMAIN,
        )
    return None


class BaselineAnalysisExpansionRuleSet:
    """Expansion rules for the baseline analysis family."""

    name = "baseline_analysis"
    domain = "baseline_analysis"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_onset_coverage_check(),
            _build_insert_padding_saturation(),
            _build_insert_normalization_clipping(),
            _build_insert_component_balance(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in (
            _diagnose_onset_coverage,
            _diagnose_padding_saturation,
            _diagnose_normalization_clipping,
            _diagnose_component_balance,
        ):
            diagnostic = fn(cdg, context)
            if diagnostic is not None:
                diagnostics.append(diagnostic)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

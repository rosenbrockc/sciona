"""Expansion rules for deterministic ML model selection pipelines.

ML pipeline skeleton topology (linear chain):

    Data Assembly -> Preprocessing -> Estimator -> Validation

Expansion insertion points:
  - Before Estimator: insert preprocessing (PowerTransformer, StandardScaler)
  - Before Estimator: insert dimensionality reduction (PCA, TruncatedSVD)
  - At Estimator: replace with recommended sklearn class (Ridge, RandomForest, etc.)
  - At Validation: force cross-validation strategy (TimeSeriesSplit, GroupKFold)

Diagnostics are driven by the 8 recommendation atoms in
``sciona.atoms.ml.model_selection.recommendations``, whose outputs
land in ``context.intermediates`` under the ``model_selection.*`` prefix.
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

_DOMAIN = "ml_model_selection"
_PREFIX = "model_selection"

_SEVERITY_MAP = {"absolute": 1.0, "high": 0.85, "medium": 0.6}


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


def _get_rec(context: ExpansionContext, key: str) -> dict | None:
    """Extract a recommendation dict from intermediates."""
    intermediates = context.intermediates or {}
    rec = intermediates.get(f"{_PREFIX}.{key}")
    if isinstance(rec, dict) and "recommendation" in rec:
        return rec
    return None


def _severity(rec: dict) -> float:
    """Map recommendation confidence to expansion severity."""
    return _SEVERITY_MAP.get(str(rec.get("confidence", "medium")), 0.5)


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_preprocessing_before_estimator() -> RewriteRule:
    """Insert a preprocessing step (PowerTransformer/StandardScaler) before estimator."""
    src = _node("src", "source", ConceptType.DATA_ASSEMBLY)
    estimator = _node("estimator", "Estimator", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[src, estimator], edges=[_edge("src", "estimator")])
    interface = CDGExport(nodes=[src, estimator], edges=[])

    preprocess = _node(
        "preprocess", "Preprocessing", ConceptType.ML_MODEL_SELECTION,
        matched_primitive="sklearn.preprocessing.StandardScaler",
        inputs=[IOSpec(name="X", type_desc="ndarray")],
        outputs=[IOSpec(name="X_transformed", type_desc="ndarray")],
        description="Apply feature-level preprocessing (scaling, power transform).",
        type_signature="ndarray -> ndarray",
    )
    rhs = CDGExport(
        nodes=[src, preprocess, estimator],
        edges=[_edge("src", "preprocess"), _edge("preprocess", "estimator")],
    )

    return RewriteRule(
        name="insert_preprocessing_before_estimator",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        priority=3,
    )


def _build_replace_estimator_from_recommendation() -> RewriteRule:
    """Replace the estimator node with the recommended sklearn class.

    Uses ``semantic_apply`` to dynamically set the ``matched_primitive``
    from the recommendation dict rather than a static RHS.
    """
    estimator = _node("estimator", "Estimator", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[estimator], edges=[])
    interface = CDGExport(nodes=[estimator], edges=[])
    rhs = CDGExport(nodes=[estimator], edges=[])

    return RewriteRule(
        name="replace_estimator_from_recommendation",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"estimator": "estimator"}, edge_map={}),
        r_morphism=Morphism(node_map={"estimator": "estimator"}, edge_map={}),
        priority=5,
    )


def _build_insert_dimensionality_reduction() -> RewriteRule:
    """Insert PCA or TruncatedSVD before the estimator."""
    src = _node("src", "source", ConceptType.DATA_ASSEMBLY)
    estimator = _node("estimator", "Estimator", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[src, estimator], edges=[_edge("src", "estimator")])
    interface = CDGExport(nodes=[src, estimator], edges=[])

    dimred = _node(
        "dimred", "Dimensionality Reduction", ConceptType.DIMENSIONALITY_REDUCTION,
        matched_primitive="sklearn.decomposition.PCA",
        inputs=[IOSpec(name="X", type_desc="ndarray")],
        outputs=[IOSpec(name="X_reduced", type_desc="ndarray")],
        description="Reduce feature dimensionality to improve conditioning.",
        type_signature="ndarray -> ndarray",
    )
    rhs = CDGExport(
        nodes=[src, dimred, estimator],
        edges=[_edge("src", "dimred"), _edge("dimred", "estimator")],
    )

    return RewriteRule(
        name="insert_dimensionality_reduction_before_estimator",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "estimator": "estimator"}, edge_map={}),
        priority=2,
    )


def _build_force_cv_strategy() -> RewriteRule:
    """Force a specific cross-validation strategy (TimeSeriesSplit, GroupKFold)."""
    validation = _node("validation", "Validation", ConceptType.ML_MODEL_SELECTION)
    lhs = CDGExport(nodes=[validation], edges=[])
    interface = CDGExport(nodes=[validation], edges=[])
    rhs = CDGExport(nodes=[validation], edges=[])

    return RewriteRule(
        name="force_cv_strategy",
        lhs=lhs, rhs=rhs, interface=interface,
        l_morphism=Morphism(node_map={"validation": "validation"}, edge_map={}),
        r_morphism=Morphism(node_map={"validation": "validation"}, edge_map={}),
        priority=4,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _diagnose_regularization(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_regularization")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    cond = intermediates.get(f"{_PREFIX}.condition_number")
    if cond is None:
        return None
    try:
        cond_val = float(cond)
    except (ValueError, TypeError):
        return None
    if cond_val <= 30.0:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Condition number {cond_val:.1f} > 30"),
        metric_name="condition_number",
        metric_value=cond_val,
        threshold=30.0,
        source_domain=_DOMAIN,
    )


def _diagnose_loss_function(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_loss_function")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    kurtosis = intermediates.get(f"{_PREFIX}.residual_kurtosis")
    if kurtosis is None:
        kurtosis = intermediates.get(f"{_PREFIX}.excess_kurtosis")
    if kurtosis is None:
        return None
    try:
        k_val = float(kurtosis)
    except (ValueError, TypeError):
        return None
    if k_val <= 1.0:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Kurtosis {k_val:.2f} > 1.0 — robust loss needed"),
        metric_name="residual_kurtosis",
        metric_value=k_val,
        threshold=1.0,
        source_domain=_DOMAIN,
    )


def _diagnose_linear_model(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_linear_model")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    di = intermediates.get(f"{_PREFIX}.dispersion_index")
    if di is None:
        return None
    try:
        di_val = float(di)
    except (ValueError, TypeError):
        return None
    if abs(di_val - 1.0) < 0.1:
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Dispersion index {di_val:.2f} deviates from 1.0"),
        metric_name="dispersion_index",
        metric_value=di_val,
        threshold=1.1,
        source_domain=_DOMAIN,
    )


def _diagnose_tree_ensemble(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_tree_ensemble")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    noise = intermediates.get(f"{_PREFIX}.noise_level")
    if noise is None:
        return None
    try:
        noise_val = float(noise)
    except (ValueError, TypeError):
        return None
    return ExpansionDiagnostic(
        rule_name="replace_estimator_from_recommendation",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Noise level {noise_val:.3f}"),
        metric_name="noise_level",
        metric_value=noise_val,
        threshold=0.5,
        source_domain=_DOMAIN,
    )


def _diagnose_preprocessing(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_preprocessing")
    if rec is None:
        return None
    # Only fire if preprocessing is actually recommended
    if rec.get("recommendation", "").lower().startswith("no preprocessing"):
        return None
    intermediates = context.intermediates or {}
    # Use max VIF or max abs skewness as the trigger metric
    vif_max = intermediates.get(f"{_PREFIX}.vif_max")
    skew_max = intermediates.get(f"{_PREFIX}.skewness_max")
    metric_name = "vif_max"
    metric_value = 0.0
    threshold = 5.0
    if vif_max is not None:
        try:
            metric_value = float(vif_max)
        except (ValueError, TypeError):
            metric_value = 0.0
    elif skew_max is not None:
        try:
            metric_value = abs(float(skew_max))
            metric_name = "skewness_max"
            threshold = 1.0
        except (ValueError, TypeError):
            metric_value = 0.0
    if metric_value <= threshold:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_preprocessing_before_estimator",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"{metric_name} {metric_value:.2f} > {threshold}"),
        metric_name=metric_name,
        metric_value=metric_value,
        threshold=threshold,
        source_domain=_DOMAIN,
    )


def _diagnose_dimensionality_reduction(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_dimensionality_reduction")
    if rec is None:
        return None
    if rec.get("recommendation", "").lower().startswith("no reduction"):
        return None
    intermediates = context.intermediates or {}
    cond = intermediates.get(f"{_PREFIX}.condition_number")
    if cond is None:
        return None
    try:
        cond_val = float(cond)
    except (ValueError, TypeError):
        return None
    if cond_val <= 30.0:
        return None
    return ExpansionDiagnostic(
        rule_name="insert_dimensionality_reduction_before_estimator",
        severity=_severity(rec),
        evidence=rec.get("reasoning", f"Condition number {cond_val:.1f} > 30 — PCA recommended"),
        metric_name="condition_number",
        metric_value=cond_val,
        threshold=30.0,
        source_domain=_DOMAIN,
    )


def _diagnose_hyperparameter_ranges(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    """Hyperparameter range recommendations don't trigger graph rewrites.

    They annotate existing nodes rather than changing topology, so we log
    but don't produce an expansion diagnostic.
    """
    return None


def _diagnose_cv_strategy(
    cdg: CDGExport, context: ExpansionContext,
) -> ExpansionDiagnostic | None:
    rec = _get_rec(context, "recommend_cv_strategy")
    if rec is None:
        return None
    intermediates = context.intermediates or {}
    is_ts = intermediates.get(f"{_PREFIX}.is_time_series")
    if is_ts is None:
        return None
    try:
        ts_flag = bool(is_ts)
    except (ValueError, TypeError):
        return None
    if not ts_flag:
        return None
    return ExpansionDiagnostic(
        rule_name="force_cv_strategy",
        severity=1.0,  # absolute confidence — time series leakage is critical
        evidence=rec.get("reasoning", "Time series detected — TimeSeriesSplit mandatory"),
        metric_name="is_time_series",
        metric_value=1.0,
        threshold=0.0,
        source_domain=_DOMAIN,
    )


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class MLModelSelectionRuleSet:
    """Expansion rules for deterministic ML model selection pipelines.

    Consumes outputs from the 8 recommendation atoms in
    ``sciona.atoms.ml.model_selection.recommendations`` and the 16
    diagnostic atoms in ``sciona.atoms.ml.model_selection.diagnostics``.
    """

    name = "ml_model_selection"
    domain = "ml_model_selection"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_preprocessing_before_estimator(),
            _build_replace_estimator_from_recommendation(),
            _build_insert_dimensionality_reduction(),
            _build_force_cv_strategy(),
        ]

    def diagnose(
        self, cdg: CDGExport, context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in [
            _diagnose_regularization,
            _diagnose_loss_function,
            _diagnose_linear_model,
            _diagnose_tree_ensemble,
            _diagnose_preprocessing,
            _diagnose_dimensionality_reduction,
            _diagnose_hyperparameter_ranges,
            _diagnose_cv_strategy,
        ]:
            d = fn(cdg, context)
            if d is not None:
                diagnostics.append(d)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

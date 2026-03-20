"""Expansion rules for the Sequential Filter family (Kalman / Particle).

Defines DPO rules and diagnostic functions that let the expansion engine
insert observability checks, divergence detection, adaptive noise estimation,
and innovation whiteness validation into Kalman/particle filter CDGs.

Kalman skeleton topology (6 nodes, bipartite predict/update):

    predict_state ──→ innovation ──→ update_state
                  ╲               ╱
    predict_cov ──→ kalman_gain ──→ update_cov

Expansion insertion points:
  - Before predict_state: observability check, adaptive noise
  - After update_cov: divergence detection, innovation whiteness

All diagnostics are pure functions of filter intermediates.
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

_DOMAIN = "sequential_filter"

# Kalman skeleton node names (as they appear in matched_primitive or name)
_PREDICT_STATE = "Predict State"
_PREDICT_COV = "Predict Covariance"
_INNOVATION = "Innovation"
_KALMAN_GAIN = "Kalman Gain"
_UPDATE_STATE = "Update State"
_UPDATE_COV = "Update Covariance"

# Particle filter node names
_PF_PREPROCESS = "Preprocess"
_PF_PREDICT = "Predict"
_PF_REWEIGHT = "Reweight"
_PF_POSTPROCESS = "Postprocess"


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


def _build_insert_observability_check() -> RewriteRule:
    """Interpose ``check_observability`` before predict_state.

    Ensures the system (F, H) is observable before running the filter.
    If unobservable modes exist, downstream state estimates will diverge
    silently on those modes.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    predict = _node(
        "predict",
        _PREDICT_STATE,
        ConceptType.SEQUENTIAL_FILTER,
    )
    lhs = CDGExport(nodes=[src, predict], edges=[_edge("src", "predict")])
    interface = CDGExport(nodes=[src, predict], edges=[])

    obs_check = _node(
        "obs_check",
        "Check Observability",
        ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="check_observability",
        inputs=[
            IOSpec(name="F", type_desc="ndarray"),
            IOSpec(name="H", type_desc="ndarray"),
            IOSpec(name="n_states", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="is_observable", type_desc="bool"),
            IOSpec(name="observability_matrix", type_desc="ndarray"),
        ],
        description="Check observability of (F, H) via rank test on O = [H; HF; HF^2; ...]",
        type_signature="ndarray, ndarray, int -> tuple[bool, ndarray]",
    )
    rhs = CDGExport(
        nodes=[src, obs_check, predict],
        edges=[
            _edge("src", "obs_check"),
            _edge("obs_check", "predict"),
        ],
    )

    return RewriteRule(
        name="insert_observability_check_before_predict",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "predict": "predict"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "predict": "predict"}, edge_map={}),
        priority=3,
    )


def _build_insert_divergence_detection() -> RewriteRule:
    """Interpose ``detect_filter_divergence`` after update_cov.

    NIS test detects whether the filter's innovation covariance S
    is consistent with the actual innovations.
    """
    update = _node(
        "update",
        _UPDATE_COV,
        ConceptType.CONJUGATE_UPDATE,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    divergence = _node(
        "divergence",
        "Detect Filter Divergence",
        ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="detect_filter_divergence",
        inputs=[
            IOSpec(name="innovations", type_desc="ndarray"),
            IOSpec(name="S_matrices", type_desc="ndarray"),
        ],
        outputs=[
            IOSpec(name="nis_values", type_desc="ndarray"),
            IOSpec(name="divergence_mask", type_desc="ndarray"),
        ],
        description="NIS chi-squared test for Kalman filter divergence.",
        type_signature="ndarray, ndarray -> tuple[ndarray, ndarray]",
    )
    rhs = CDGExport(
        nodes=[update, divergence, sink],
        edges=[
            _edge("update", "divergence"),
            _edge("divergence", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_divergence_detection_after_update",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=2,
    )


def _build_insert_adaptive_noise() -> RewriteRule:
    """Interpose ``adapt_process_noise`` before predict_cov.

    When the assumed Q doesn't match reality, the filter's covariance
    predictions are wrong, leading to suboptimal gain and potential
    divergence.  Adaptive Q estimation corrects this from innovations.
    """
    src = _node("src", "source", ConceptType.CUSTOM)
    predict_cov = _node(
        "predict_cov",
        _PREDICT_COV,
        ConceptType.SEQUENTIAL_FILTER,
    )
    lhs = CDGExport(
        nodes=[src, predict_cov],
        edges=[_edge("src", "predict_cov")],
    )
    interface = CDGExport(nodes=[src, predict_cov], edges=[])

    adapt = _node(
        "adapt_q",
        "Adapt Process Noise",
        ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="adapt_process_noise",
        inputs=[
            IOSpec(name="innovations", type_desc="ndarray"),
            IOSpec(name="K_matrices", type_desc="ndarray"),
            IOSpec(name="Q_prior", type_desc="ndarray"),
        ],
        outputs=[IOSpec(name="Q_adapted", type_desc="ndarray")],
        description="Robbins-Monro adaptive Q estimation from innovations.",
        type_signature="ndarray, ndarray, ndarray -> ndarray",
    )
    rhs = CDGExport(
        nodes=[src, adapt, predict_cov],
        edges=[
            _edge("src", "adapt_q"),
            _edge("adapt_q", "predict_cov"),
        ],
    )

    return RewriteRule(
        name="insert_adaptive_noise_before_predict_cov",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={"src": "src", "predict_cov": "predict_cov"}, edge_map={}
        ),
        r_morphism=Morphism(
            node_map={"src": "src", "predict_cov": "predict_cov"}, edge_map={}
        ),
        priority=1,
    )


def _build_insert_innovation_whiteness() -> RewriteRule:
    """Interpose ``validate_innovation_whiteness`` after update_state.

    White (uncorrelated) innovations indicate a correctly tuned filter.
    Non-white innovations signal model mismatch or incorrect noise params.
    """
    update = _node(
        "update",
        _UPDATE_STATE,
        ConceptType.CONJUGATE_UPDATE,
    )
    sink = _node("sink", "sink", ConceptType.CUSTOM)
    lhs = CDGExport(nodes=[update, sink], edges=[_edge("update", "sink")])
    interface = CDGExport(nodes=[update, sink], edges=[])

    whiteness = _node(
        "whiteness",
        "Validate Innovation Whiteness",
        ConceptType.SEQUENTIAL_FILTER,
        matched_primitive="validate_innovation_whiteness",
        inputs=[
            IOSpec(name="innovations", type_desc="ndarray"),
            IOSpec(name="max_lag", type_desc="int"),
        ],
        outputs=[
            IOSpec(name="autocorrelation", type_desc="ndarray"),
            IOSpec(name="is_white", type_desc="bool"),
        ],
        description="Autocorrelation test on innovations for model adequacy.",
        type_signature="ndarray, int -> tuple[ndarray, bool]",
    )
    rhs = CDGExport(
        nodes=[update, whiteness, sink],
        edges=[
            _edge("update", "whiteness"),
            _edge("whiteness", "sink"),
        ],
    )

    return RewriteRule(
        name="insert_innovation_whiteness_after_update",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        r_morphism=Morphism(node_map={"update": "update", "sink": "sink"}, edge_map={}),
        priority=1,
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_observability(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect rank-deficient observability matrix."""
    data = context.signal_data or {}
    F = data.get("F")
    H = data.get("H")
    n_states = data.get("n_states")

    if F is None or H is None or n_states is None:
        return None

    try:
        F = np.atleast_2d(np.asarray(F, dtype=np.float64))
        H = np.atleast_2d(np.asarray(H, dtype=np.float64))
        n = int(n_states)
    except (ValueError, TypeError):
        return None

    if F.shape[0] != F.shape[1] or F.shape[0] != n:
        return None

    rows = [H]
    HFk = H.copy()
    for _ in range(n - 1):
        HFk = HFk @ F
        rows.append(HFk)
    O = np.vstack(rows)
    rank = int(np.linalg.matrix_rank(O))
    deficit = n - rank

    if deficit > 0:
        return ExpansionDiagnostic(
            rule_name="insert_observability_check_before_predict",
            severity=min(1.0, deficit / n),
            evidence=f"Observability rank deficit: {deficit} of {n} states unobservable",
            metric_name="observability_rank_deficit",
            metric_value=float(deficit),
            threshold=0.0,
            source_domain=_DOMAIN,
        )
    return None


def _diagnose_innovation_non_whiteness(
    cdg: CDGExport, context: ExpansionContext
) -> list[ExpansionDiagnostic]:
    """Detect non-white innovations indicating model mismatch."""
    innovations = (context.intermediates or {}).get("innovations")
    if innovations is None:
        return []

    innovations = np.asarray(innovations, dtype=np.float64)
    if innovations.ndim > 1:
        innovations = np.array(
            [np.linalg.norm(innovations[i]) for i in range(len(innovations))]
        )

    n = len(innovations)
    if n < 12:
        return []

    max_lag = min(10, n // 2)
    centered = innovations - np.mean(innovations)
    var = float(np.var(centered))
    if var < 1e-15:
        return []

    acf = np.zeros(max_lag)
    for lag in range(1, max_lag + 1):
        acf[lag - 1] = float(np.mean(centered[:-lag] * centered[lag:])) / var

    max_acf = float(np.max(np.abs(acf)))
    bound = 1.96 / np.sqrt(n)

    if max_acf <= bound:
        return []

    diagnostics: list[ExpansionDiagnostic] = []
    diagnostics.append(
        ExpansionDiagnostic(
            rule_name="insert_innovation_whiteness_after_update",
            severity=min(1.0, max_acf / (3.0 * bound)),
            evidence=(
                f"Max innovation ACF={max_acf:.3f} exceeds 95% bound={bound:.3f} "
                f"at {n} samples"
            ),
            metric_name="max_innovation_acf",
            metric_value=max_acf,
            threshold=bound,
            source_domain=_DOMAIN,
        )
    )
    # If innovations are strongly non-white, also recommend adaptive noise
    if max_acf > 2.0 * bound:
        diagnostics.append(
            ExpansionDiagnostic(
                rule_name="insert_adaptive_noise_before_predict_cov",
                severity=min(1.0, max_acf / (5.0 * bound)),
                evidence=(
                    f"Strongly non-white innovations (ACF={max_acf:.3f} >> bound={bound:.3f}) "
                    f"suggest process noise mismatch"
                ),
                metric_name="max_innovation_acf",
                metric_value=max_acf,
                threshold=2.0 * bound,
                source_domain=_DOMAIN,
            )
        )
    return diagnostics


def _diagnose_nis_divergence(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect filter divergence via NIS chi-squared test."""
    intermediates = context.intermediates or {}
    innovations = intermediates.get("innovations")
    S_matrices = intermediates.get("S_matrices")

    if innovations is None or S_matrices is None:
        return None

    innovations = np.asarray(innovations, dtype=np.float64)
    S_matrices = np.asarray(S_matrices, dtype=np.float64)

    n = len(innovations)
    if n < 5:
        return None

    if innovations.ndim == 1:
        innovations = innovations.reshape(-1, 1)

    m = innovations.shape[1]
    # Chi-squared 95th percentile
    chi2_95_table = {1: 3.841, 2: 5.991, 3: 7.815}
    if m in chi2_95_table:
        chi2_95 = chi2_95_table[m]
    else:
        p = 2.0 / (9.0 * m)
        chi2_95 = m * (1.0 - p + 1.645 * np.sqrt(p)) ** 3

    exceed_count = 0
    for k in range(n):
        y = innovations[k]
        S = S_matrices[k] if S_matrices.ndim == 3 else S_matrices
        try:
            S_inv = np.linalg.solve(S, np.eye(S.shape[0]))
            nis = float(y @ S_inv @ y)
        except np.linalg.LinAlgError:
            nis = float("inf")
        if nis > chi2_95:
            exceed_count += 1

    exceed_frac = exceed_count / n
    threshold = 0.10  # more than 10% exceeding → divergence

    if exceed_frac > threshold:
        return ExpansionDiagnostic(
            rule_name="insert_divergence_detection_after_update",
            severity=min(1.0, exceed_frac / 0.4),
            evidence=(
                f"{exceed_frac:.1%} of NIS values exceed chi-squared({m}) 95th percentile "
                f"(>{threshold:.0%} threshold)"
            ),
            metric_name="nis_exceed_fraction",
            metric_value=exceed_frac,
            threshold=threshold,
            source_domain=_DOMAIN,
        )
    return None


# ---------------------------------------------------------------------------
# Rule set
# ---------------------------------------------------------------------------


class SequentialFilterExpansionRuleSet:
    """Expansion rules for Kalman and particle filter pipelines."""

    name = "sequential_filter"
    domain = "sequential_filter"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_observability_check(),
            _build_insert_divergence_detection(),
            _build_insert_adaptive_noise(),
            _build_insert_innovation_whiteness(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        diagnostics: list[ExpansionDiagnostic] = []

        obs = _diagnose_observability(cdg, context)
        if obs is not None:
            diagnostics.append(obs)

        diagnostics.extend(_diagnose_innovation_non_whiteness(cdg, context))

        nis = _diagnose_nis_divergence(cdg, context)
        if nis is not None:
            diagnostics.append(nis)

        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

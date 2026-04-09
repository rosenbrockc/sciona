"""Expansion rules for the signal → event → rate family.

Defines DPO rules and diagnostic functions that let the expansion engine
insert additional processing stages (SQI gating, jump removal, outlier
rejection) into an existing filter → detect → rate CDG.

All diagnostics are pure functions of signal data / intermediate results.
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
from sciona.architect.semantic_rewrites import build_boundary_interposition_callback
from sciona.principal.expansion import (
    ExpansionContext,
    ExpansionDiagnostic,
)
from sciona.principal.expansion_assets import (
    ExpansionFamilyAsset,
    expansion_asset_summary,
    load_local_expansion_assets_by_family,
)
from sciona.principal.runtime_context import summarize_waveform

logger = logging.getLogger(__name__)

_DOMAIN = "signal_event_rate"


def _signal_event_rate_asset() -> ExpansionFamilyAsset | None:
    return load_local_expansion_assets_by_family().get("signal_event_rate")


def _diag_asset_fields(rule_name: str) -> dict[str, object]:
    asset = _signal_event_rate_asset()
    if asset is None:
        return {}
    operation = asset.operation(rule_name)
    if operation is None:
        return {}
    summary = expansion_asset_summary(asset, operation)
    return {
        key: summary[key]
        for key in summary
        if key.startswith("asset_")
    }


# ---------------------------------------------------------------------------
# Node / edge helpers for rule construction
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
    type_desc: str = "np.ndarray",
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
# Standard IOSpec fragments
# ---------------------------------------------------------------------------

_SIG_IN = IOSpec(name="signal", type_desc="np.ndarray")
_RATE_IN = IOSpec(name="sampling_rate", type_desc="float")
_SIG_OUT = IOSpec(name="signal", type_desc="np.ndarray")
_MASK_OUT = IOSpec(name="quality_mask", type_desc="np.ndarray")
_EVENTS_IN = IOSpec(name="events", type_desc="np.ndarray")
_EVENTS_OUT = IOSpec(name="events", type_desc="np.ndarray")
_FILTERED_IN = IOSpec(name="filtered", type_desc="np.ndarray")


# ---------------------------------------------------------------------------
# DPO rule builders
# ---------------------------------------------------------------------------


def _build_insert_jump_removal_before_filter() -> RewriteRule:
    """Interpose ``remove_signal_jumps`` before the bandpass filter."""
    # L: [src] ---> [filter]
    src_l = _node("src", "source", ConceptType.CUSTOM)
    filt_l = _node(
        "filt",
        "filter",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="filter_signal_for_detection",
    )
    lhs = CDGExport(
        nodes=[src_l, filt_l],
        edges=[_edge("src", "filt", "signal", "signal")],
    )

    # K: [src], [filt]  (no edge — the edge is consumed)
    interface = CDGExport(nodes=[src_l, filt_l], edges=[])

    # R: [src] -> [jump_removal] -> [filt]
    jump_r = _node(
        "jump",
        "Remove Signal Jumps",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="remove_signal_jumps",
        inputs=[_SIG_IN, _RATE_IN],
        outputs=[_SIG_OUT],
        description="Remove step discontinuities from raw signal.",
        type_signature="np.ndarray, float -> np.ndarray",
    )
    rhs = CDGExport(
        nodes=[src_l, jump_r, filt_l],
        edges=[
            _edge("src", "jump", "signal", "signal"),
            _edge("jump", "filt", "signal", "signal"),
        ],
    )

    return RewriteRule(
        name="insert_jump_removal_before_filter",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "filt": "filt"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "filt": "filt"}, edge_map={}),
        priority=2,
        anchor_type="filter_signal_for_detection",
        semantic_apply=build_boundary_interposition_callback(
            target_primitive="filter_signal_for_detection",
            boundary_input_name="signal",
            insert_node=jump_r,
            target_input_name="signal",
            insert_output_name="signal",
        ),
    )


def _build_insert_sqi_before_filter() -> RewriteRule:
    """Interpose ``assess_signal_quality`` before the bandpass filter."""
    src_l = _node("src", "source", ConceptType.CUSTOM)
    filt_l = _node(
        "filt",
        "filter",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="filter_signal_for_detection",
    )
    lhs = CDGExport(
        nodes=[src_l, filt_l],
        edges=[_edge("src", "filt", "signal", "signal")],
    )
    interface = CDGExport(nodes=[src_l, filt_l], edges=[])

    sqi_r = _node(
        "sqi",
        "Assess Signal Quality",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="assess_signal_quality",
        inputs=[_SIG_IN, _RATE_IN],
        outputs=[_SIG_OUT, _MASK_OUT],
        description="Compute per-window signal quality mask using kurtosis.",
        type_signature="np.ndarray, float -> tuple[np.ndarray, np.ndarray]",
    )
    rhs = CDGExport(
        nodes=[src_l, sqi_r, filt_l],
        edges=[
            _edge("src", "sqi", "signal", "signal"),
            _edge("sqi", "filt", "signal", "signal"),
        ],
    )

    return RewriteRule(
        name="insert_sqi_before_filter",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(node_map={"src": "src", "filt": "filt"}, edge_map={}),
        r_morphism=Morphism(node_map={"src": "src", "filt": "filt"}, edge_map={}),
        priority=1,
        anchor_type="filter_signal_for_detection",
        semantic_apply=build_boundary_interposition_callback(
            target_primitive="filter_signal_for_detection",
            boundary_input_name="signal",
            insert_node=sqi_r,
            target_input_name="signal",
            insert_output_name="signal",
        ),
    )


def _build_insert_outlier_rejection_after_detection() -> RewriteRule:
    """Interpose ``reject_outlier_intervals`` between detection and rate."""
    detect_l = _node(
        "detect",
        "detect_peaks",
        ConceptType.DATA_EXTRACTION,
        matched_primitive="detect_peaks_in_signal",
    )
    rate_l = _node(
        "rate",
        "compute_rate",
        ConceptType.ANALYSIS,
        matched_primitive="compute_event_rate",
    )
    lhs = CDGExport(
        nodes=[detect_l, rate_l],
        edges=[_edge("detect", "rate", "events", "events")],
    )
    interface = CDGExport(nodes=[detect_l, rate_l], edges=[])

    reject_r = _node(
        "reject",
        "Reject Outlier Intervals",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="ageoa.biosppy.ecg.reject_outlier_intervals",
        inputs=[_EVENTS_IN, _RATE_IN],
        outputs=[_EVENTS_OUT],
        description="Remove events creating physiologically implausible intervals.",
        type_signature="np.ndarray, float -> np.ndarray",
    )
    rhs = CDGExport(
        nodes=[detect_l, reject_r, rate_l],
        edges=[
            _edge("detect", "reject", "events", "events"),
            _edge("reject", "rate", "events", "events"),
        ],
    )

    return RewriteRule(
        name="insert_outlier_rejection_after_detection",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={"detect": "detect", "rate": "rate"}, edge_map={}
        ),
        r_morphism=Morphism(
            node_map={"detect": "detect", "rate": "rate"}, edge_map={}
        ),
        priority=3,
        anchor_type="detect_peaks_in_signal",
    )


def _build_insert_peak_correction_after_detection() -> RewriteRule:
    """Insert ``peak_correction`` before rate estimation using the filtered signal."""
    filt_l = _node(
        "filt",
        "filter",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="filter_signal_for_detection",
    )
    detect_l = _node(
        "detect",
        "detect_peaks",
        ConceptType.DATA_EXTRACTION,
        matched_primitive="detect_peaks_in_signal",
    )
    rate_l = _node(
        "rate",
        "compute_rate",
        ConceptType.ANALYSIS,
        matched_primitive="compute_event_rate",
    )
    lhs = CDGExport(
        nodes=[filt_l, detect_l, rate_l],
        edges=[
            _edge("filt", "detect", "signal", "signal"),
            _edge("detect", "rate", "events", "events"),
        ],
    )
    interface = CDGExport(nodes=[filt_l, detect_l, rate_l], edges=[])

    correct_r = _node(
        "correct",
        "Correct Detected Events",
        ConceptType.DATA_EXTRACTION,
        matched_primitive="peak_correction",
        inputs=[_FILTERED_IN, _EVENTS_IN, _RATE_IN],
        outputs=[_EVENTS_OUT],
        description="Refine detected event locations against the conditioned waveform.",
        type_signature="np.ndarray, np.ndarray, float -> np.ndarray",
    )
    rhs = CDGExport(
        nodes=[filt_l, detect_l, correct_r, rate_l],
        edges=[
            _edge("filt", "detect", "signal", "signal"),
            _edge("filt", "correct", "signal", "filtered"),
            _edge("detect", "correct", "events", "events"),
            _edge("correct", "rate", "events", "events"),
        ],
    )

    return RewriteRule(
        name="insert_peak_correction_after_detection",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={"filt": "filt", "detect": "detect", "rate": "rate"},
            edge_map={},
        ),
        r_morphism=Morphism(
            node_map={"filt": "filt", "detect": "detect", "rate": "rate"},
            edge_map={},
        ),
        priority=4,
        anchor_type="detect_peaks_in_signal",
    )


def _build_insert_outlier_rejection_after_detection_smoothed() -> RewriteRule:
    """Same as above but targets compute_event_rate_smoothed."""
    detect_l = _node(
        "detect",
        "detect_peaks",
        ConceptType.DATA_EXTRACTION,
        matched_primitive="detect_peaks_in_signal",
    )
    rate_l = _node(
        "rate",
        "compute_rate_smoothed",
        ConceptType.ANALYSIS,
        matched_primitive="compute_event_rate_smoothed",
    )
    lhs = CDGExport(
        nodes=[detect_l, rate_l],
        edges=[_edge("detect", "rate", "events", "events")],
    )
    interface = CDGExport(nodes=[detect_l, rate_l], edges=[])

    reject_r = _node(
        "reject",
        "Reject Outlier Intervals",
        ConceptType.SIGNAL_FILTER,
        matched_primitive="ageoa.biosppy.ecg.reject_outlier_intervals",
        inputs=[_EVENTS_IN, _RATE_IN],
        outputs=[_EVENTS_OUT],
        description="Remove events creating physiologically implausible intervals.",
        type_signature="np.ndarray, float -> np.ndarray",
    )
    rhs = CDGExport(
        nodes=[detect_l, reject_r, rate_l],
        edges=[
            _edge("detect", "reject", "events", "events"),
            _edge("reject", "rate", "events", "events"),
        ],
    )

    return RewriteRule(
        name="insert_outlier_rejection_after_detection_smoothed",
        lhs=lhs,
        rhs=rhs,
        interface=interface,
        l_morphism=Morphism(
            node_map={"detect": "detect", "rate": "rate"}, edge_map={}
        ),
        r_morphism=Morphism(
            node_map={"detect": "detect", "rate": "rate"}, edge_map={}
        ),
        priority=3,
        anchor_type="detect_peaks_in_signal",
    )


# ---------------------------------------------------------------------------
# Diagnostics (pure, deterministic)
# ---------------------------------------------------------------------------


def _diagnose_jump_discontinuities(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect signal jumps that warrant pre-filter dejumping."""
    signal = (context.signal_data or {}).get("signal")
    jump_count: int | None = None
    if signal is not None:
        values = np.asarray(signal, dtype=np.float64).reshape(-1)
        if values.size >= 10:
            jump_count = int(
                float(summarize_waveform(values).get("discontinuity_count", 0.0))
            )
    if jump_count is None and isinstance(context.runtime_evidence, dict):
        telemetry_summary = context.runtime_evidence.get("telemetry_summary", {})
        if isinstance(telemetry_summary, dict):
            signal_summary = telemetry_summary.get("signal", {})
            if isinstance(signal_summary, dict):
                try:
                    jump_count = int(float(signal_summary.get("discontinuity_count")))
                except (TypeError, ValueError):
                    jump_count = None
    if jump_count is None:
        return None
    threshold = 3

    if jump_count > threshold:
        return ExpansionDiagnostic(
            rule_name="insert_jump_removal_before_filter",
            severity=min(1.0, jump_count / 20.0),
            evidence=f"{jump_count} discontinuities detected (>{threshold} threshold)",
            metric_name="jump_discontinuity_count",
            metric_value=float(jump_count),
            threshold=float(threshold),
            source_domain=_DOMAIN,
            **_diag_asset_fields("insert_jump_removal_before_filter"),
        )
    return None


def _diagnose_signal_quality_variance(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect non-stationary signal quality requiring SQI gating."""
    signal = (context.signal_data or {}).get("signal")
    sampling_rate = (context.signal_data or {}).get("sampling_rate")
    if signal is None or sampling_rate is None:
        return None

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    rate = float(sampling_rate)
    if values.size < 20 or rate <= 0:
        return None

    window = max(1, int(round(10.0 * rate)))  # 10-second windows
    kurtosis_values: list[float] = []
    for start in range(0, values.size, window):
        seg = values[start : start + window]
        if seg.size < 4:
            continue
        centered = seg - np.mean(seg)
        std = float(np.std(centered))
        if std < 1e-10:
            kurtosis_values.append(0.0)
            continue
        kurt = float(np.mean((centered / std) ** 4)) - 3.0
        kurtosis_values.append(kurt)

    if len(kurtosis_values) < 2:
        return None

    kurt_arr = np.array(kurtosis_values)
    kurt_cv = float(np.std(kurt_arr) / max(abs(np.mean(kurt_arr)), 1e-10))
    threshold = 1.5  # coefficient of variation of per-window kurtosis

    if kurt_cv > threshold:
        return ExpansionDiagnostic(
            rule_name="insert_sqi_before_filter",
            severity=min(1.0, kurt_cv / 5.0),
            evidence=(
                f"Kurtosis CV={kurt_cv:.2f} across {len(kurtosis_values)} windows "
                f"(>{threshold:.1f} threshold)"
            ),
            metric_name="signal_quality_kurtosis_cv",
            metric_value=kurt_cv,
            threshold=threshold,
            source_domain=_DOMAIN,
            **_diag_asset_fields("insert_sqi_before_filter"),
        )
    return None


def _diagnose_interval_outlier_fraction(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect excessive outlier intervals after peak detection."""
    events = (context.intermediates or {}).get("events")
    outlier_frac: float | None = None
    if events is not None:
        idx = np.asarray(events, dtype=np.float64).reshape(-1)
        if idx.size >= 5:
            intervals = np.diff(np.sort(idx))
            intervals = intervals[intervals > 0]
            if intervals.size >= 3:
                median_ivl = float(np.median(intervals))
                mad_ivl = float(np.median(np.abs(intervals - median_ivl)))
                if mad_ivl >= 1e-10:
                    lo = median_ivl - 3.0 * mad_ivl
                    hi = median_ivl + 3.0 * mad_ivl
                    outlier_frac = float(np.mean((intervals < lo) | (intervals > hi)))
    if outlier_frac is None:
        telemetry_summary = {}
        if isinstance(context.runtime_evidence, dict):
            telemetry_summary = context.runtime_evidence.get("telemetry_summary", {})
        if isinstance(telemetry_summary, dict):
            event_summary = telemetry_summary.get("events", {})
            if isinstance(event_summary, dict):
                try:
                    outlier_frac = float(event_summary.get("outlier_fraction"))
                except (TypeError, ValueError):
                    outlier_frac = None
    if outlier_frac is None:
        return None
    threshold = 0.08  # stronger instability warrants lossy cleanup

    if outlier_frac > threshold:
        # Pick the right rule depending on which rate primitive is present
        rule_name = "insert_outlier_rejection_after_detection"
        for node in cdg.nodes:
            if node.matched_primitive == "compute_event_rate_smoothed":
                rule_name = "insert_outlier_rejection_after_detection_smoothed"
                break

        return ExpansionDiagnostic(
            rule_name=rule_name,
            severity=min(1.0, outlier_frac / 0.5),
            evidence=(
                f"{outlier_frac:.1%} of inter-event intervals are outliers "
                f"(>{threshold:.0%} threshold)"
            ),
            metric_name="interval_outlier_fraction",
            metric_value=outlier_frac,
            threshold=threshold,
            source_domain=_DOMAIN,
            **_diag_asset_fields(rule_name),
        )
    return None


def _diagnose_peak_correction_need(
    cdg: CDGExport, context: ExpansionContext
) -> ExpansionDiagnostic | None:
    """Detect moderate event-stream irregularity that warrants peak correction."""
    if any(str(node.matched_primitive or "") == "peak_correction" for node in cdg.nodes):
        return None

    outlier_frac: float | None = None
    telemetry_summary = {}
    if isinstance(context.runtime_evidence, dict):
        telemetry_summary = context.runtime_evidence.get("telemetry_summary", {})
    if isinstance(telemetry_summary, dict):
        event_summary = telemetry_summary.get("events", {})
        if isinstance(event_summary, dict):
            try:
                outlier_frac = float(event_summary.get("outlier_fraction"))
            except (TypeError, ValueError):
                outlier_frac = None

    if outlier_frac is None:
        events = (context.intermediates or {}).get("events")
        if events is not None:
            idx = np.asarray(events, dtype=np.float64).reshape(-1)
            if idx.size >= 5:
                intervals = np.diff(np.sort(idx))
                intervals = intervals[intervals > 0]
                if intervals.size >= 3:
                    median_ivl = float(np.median(intervals))
                    mad_ivl = float(np.median(np.abs(intervals - median_ivl)))
                    if mad_ivl >= 1e-10:
                        lo = median_ivl - 3.0 * mad_ivl
                        hi = median_ivl + 3.0 * mad_ivl
                        outlier_frac = float(np.mean((intervals < lo) | (intervals > hi)))

    if outlier_frac is None:
        return None

    threshold = 0.05
    escalation_threshold = 0.08
    if threshold < outlier_frac < escalation_threshold:
        return ExpansionDiagnostic(
            rule_name="insert_peak_correction_after_detection",
            severity=min(1.0, max(0.35, outlier_frac / 0.2)),
            evidence=(
                f"{outlier_frac:.1%} of inter-event intervals are irregular "
                f"(>{threshold:.0%} threshold), suggesting event timing drift"
            ),
            metric_name="interval_outlier_fraction",
            metric_value=outlier_frac,
            threshold=threshold,
            source_domain=_DOMAIN,
            **_diag_asset_fields("insert_peak_correction_after_detection"),
        )
    return None


# ---------------------------------------------------------------------------
# Rule set implementation
# ---------------------------------------------------------------------------


class SignalEventRateExpansionRuleSet:
    """Expansion rules for signal → event → rate pipelines."""

    name = "signal_event_rate"
    domain = "signal_processing"

    def __init__(self) -> None:
        self._rules = [
            _build_insert_jump_removal_before_filter(),
            _build_insert_sqi_before_filter(),
            _build_insert_peak_correction_after_detection(),
            _build_insert_outlier_rejection_after_detection(),
            _build_insert_outlier_rejection_after_detection_smoothed(),
        ]

    def diagnose(
        self,
        cdg: CDGExport,
        context: ExpansionContext,
    ) -> list[ExpansionDiagnostic]:
        """Run all signal-processing diagnostics.

        Diagnostics that don't find the data they need in *context*
        return ``None`` and are silently skipped — this is how
        cross-domain irrelevance is expressed without a ``matches()`` gate.
        """
        diagnostics: list[ExpansionDiagnostic] = []
        for fn in (
            _diagnose_jump_discontinuities,
            _diagnose_signal_quality_variance,
            _diagnose_peak_correction_need,
            _diagnose_interval_outlier_fraction,
        ):
            diag = fn(cdg, context)
            if diag is not None:
                diagnostics.append(diag)
        return diagnostics

    def rules(self) -> list[RewriteRule]:
        return list(self._rules)

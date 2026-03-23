"""Tests for the Signal Detect Measure expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.signal_detect_measure import SignalDetectMeasureExpansionRuleSet
from sciona.expansion_atoms.runtime_signal_detect_measure import (
    estimate_snr, analyze_peak_threshold_sensitivity,
    check_event_rate_stationarity, estimate_false_positive_rate,
)


def _node(nid, name, concept=ConceptType.CUSTOM, primitive=None):
    return AlgorithmicNode(
        node_id=nid, name=name, description=name, concept_type=concept,
        status=NodeStatus.ATOMIC, matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="ndarray")],
        outputs=[IOSpec(name="out", type_desc="ndarray")],
        type_signature=f"{name} -> r",
    )

def _edge(src, tgt):
    return DependencyEdge(source_id=src, target_id=tgt, output_name="out", input_name="in", source_type="ndarray", target_type="ndarray")

def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})

def _signal_detect_cdg():
    return _cdg(
        [_node("src", "Source"), _node("fs", "Filter Signal For Detection", ConceptType.SIGNAL_FILTER),
         _node("dp", "Detect Peaks In Signal", ConceptType.DATA_EXTRACTION),
         _node("cr", "Compute Event Rate", ConceptType.ANALYSIS),
         _node("out", "Output")],
        [_edge("src", "fs"), _edge("fs", "dp"), _edge("dp", "cr"), _edge("cr", "out")],
    )


class TestEstimateSNR:
    def test_high_snr(self):
        # Provide explicit noise floor for reliable test
        rng = np.random.RandomState(42)
        signal = np.sin(np.linspace(0, 10 * np.pi, 1000)) * 10 + rng.randn(1000) * 0.01
        snr_db, sufficient = estimate_snr(signal, noise_floor=0.01 ** 2)
        assert sufficient

    def test_low_snr(self):
        noise = np.random.RandomState(42).randn(1000)
        snr_db, sufficient = estimate_snr(noise)
        assert isinstance(snr_db, float)

    def test_short_signal(self):
        snr_db, sufficient = estimate_snr(np.array([1.0]))
        assert not sufficient


class TestAnalyzePeakThresholdSensitivity:
    def test_stable(self):
        peaks = np.array([10, 20, 30, 40, 50], dtype=float)
        sens, stable = analyze_peak_threshold_sensitivity(peaks, 5.0)
        assert stable

    def test_sensitive(self):
        peaks = np.array([9.9, 10.0, 10.1, 10.05, 9.95], dtype=float)
        sens, stable = analyze_peak_threshold_sensitivity(peaks, 10.0)
        assert not stable

    def test_empty(self):
        sens, stable = analyze_peak_threshold_sensitivity(np.array([]), 1.0)
        assert stable


class TestCheckEventRateStationarity:
    def test_stationary(self):
        events = np.linspace(0, 100, 200)
        cv, stationary = check_event_rate_stationarity(events)
        assert stationary

    def test_non_stationary(self):
        # Cluster events in first half
        events = np.concatenate([np.linspace(0, 10, 100), np.array([90, 95, 100])])
        cv, stationary = check_event_rate_stationarity(events)
        assert not stationary

    def test_single_event(self):
        cv, stationary = check_event_rate_stationarity(np.array([5.0]))
        assert stationary


class TestEstimateFalsePositiveRate:
    def test_reliable(self):
        amplitudes = np.array([100, 200, 300], dtype=float)
        fpr, reliable = estimate_false_positive_rate(amplitudes, 1.0, 10.0)
        assert reliable

    def test_unreliable(self):
        amplitudes = np.array([10.5, 10.8, 11.0, 11.2], dtype=float)
        fpr, reliable = estimate_false_positive_rate(amplitudes, 1.0, 10.0)
        assert not reliable

    def test_empty(self):
        fpr, reliable = estimate_false_positive_rate(np.array([]), 1.0, 10.0)
        assert reliable


class TestSignalDetectMeasureRules:
    def _get_rules(self):
        return {r.name: r for r in SignalDetectMeasureExpansionRuleSet().rules()}

    def test_snr_estimation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_snr_estimation_before_filter"], _signal_detect_cdg())
        assert not result.is_failure
        assert "estimate_snr" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_threshold_sensitivity_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_threshold_sensitivity_after_filter"], _signal_detect_cdg())
        assert not result.is_failure

    def test_rate_stationarity_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_rate_stationarity_after_detect"], _signal_detect_cdg())
        assert not result.is_failure

    def test_false_positive_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_false_positive_estimation_after_rate"], _signal_detect_cdg())
        assert not result.is_failure


class TestSignalDetectMeasureDiagnostics:
    def test_diagnose_low_snr(self):
        diags = SignalDetectMeasureExpansionRuleSet().diagnose(_signal_detect_cdg(), ExpansionContext(intermediates={"snr_db": 5.0}))
        assert "insert_snr_estimation_before_filter" in {d.rule_name for d in diags}

    def test_high_snr_no_trigger(self):
        diags = SignalDetectMeasureExpansionRuleSet().diagnose(_signal_detect_cdg(), ExpansionContext(intermediates={"snr_db": 20.0}))
        assert not [d for d in diags if d.rule_name == "insert_snr_estimation_before_filter"]

    def test_diagnose_sensitivity(self):
        diags = SignalDetectMeasureExpansionRuleSet().diagnose(_signal_detect_cdg(), ExpansionContext(intermediates={"threshold_sensitivity": 0.4}))
        assert "insert_threshold_sensitivity_after_filter" in {d.rule_name for d in diags}

    def test_diagnose_stationarity(self):
        diags = SignalDetectMeasureExpansionRuleSet().diagnose(_signal_detect_cdg(), ExpansionContext(intermediates={"event_rate_cv": 0.8}))
        assert "insert_rate_stationarity_after_detect" in {d.rule_name for d in diags}

    def test_diagnose_fpr(self):
        diags = SignalDetectMeasureExpansionRuleSet().diagnose(_signal_detect_cdg(), ExpansionContext(intermediates={"false_positive_rate": 0.15}))
        assert "insert_false_positive_estimation_after_rate" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert SignalDetectMeasureExpansionRuleSet().diagnose(_signal_detect_cdg(), ExpansionContext()) == []


class TestSignalDetectMeasureIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([SignalDetectMeasureExpansionRuleSet()]).expand(
            _signal_detect_cdg(), ExpansionContext(intermediates={"snr_db": 3.0, "false_positive_rate": 0.2}))
        assert result.expanded

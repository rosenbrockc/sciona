"""Tests for the Signal Filter expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.signal_filter import SignalFilterExpansionRuleSet
from sciona.expansion_atoms.runtime_signal_filter import (
    analyze_pole_stability, measure_passband_ripple,
    analyze_group_delay_variation, detect_transient_response,
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

def _signal_filter_cdg():
    return _cdg(
        [_node("src", "Source"), _node("df", "Design Filter", ConceptType.SIGNAL_FILTER),
         _node("vs", "Validate Stability", ConceptType.SIGNAL_FILTER),
         _node("af", "Apply Filter", ConceptType.SIGNAL_FILTER),
         _node("fr", "Frequency Response", ConceptType.SIGNAL_FILTER),
         _node("out", "Output")],
        [_edge("src", "df"), _edge("df", "vs"), _edge("vs", "af"), _edge("vs", "fr"),
         _edge("af", "out"), _edge("fr", "out")],
    )


class TestAnalyzePoleStability:
    def test_stable(self):
        poles = np.array([0.5 + 0.3j, 0.5 - 0.3j])
        mag, stable = analyze_pole_stability(poles)
        assert stable
        assert mag < 1.0

    def test_unstable(self):
        poles = np.array([1.1 + 0j, 0.5 + 0j])
        mag, stable = analyze_pole_stability(poles)
        assert not stable
        assert mag > 1.0

    def test_empty(self):
        mag, stable = analyze_pole_stability(np.array([], dtype=complex))
        assert stable


class TestMeasurePassbandRipple:
    def test_flat(self):
        resp = np.zeros(100)
        mask = np.ones(100, dtype=bool)
        ripple, ok = measure_passband_ripple(resp, mask)
        assert ok
        assert ripple == 0.0

    def test_ripply(self):
        resp = np.array([0, 2, 0, 2, 0, 2], dtype=float)
        mask = np.ones(6, dtype=bool)
        ripple, ok = measure_passband_ripple(resp, mask)
        assert not ok
        assert ripple == 2.0

    def test_empty(self):
        ripple, ok = measure_passband_ripple(np.array([]), np.array([], dtype=bool))
        assert ok


class TestAnalyzeGroupDelayVariation:
    def test_constant(self):
        gd = np.ones(100) * 5.0
        var, linear = analyze_group_delay_variation(gd)
        assert linear
        assert var == 0.0

    def test_variable(self):
        gd = np.linspace(0, 10, 100)
        var, linear = analyze_group_delay_variation(gd)
        assert not linear
        assert var > 1.0

    def test_single(self):
        var, linear = analyze_group_delay_variation(np.array([5.0]))
        assert linear


class TestDetectTransientResponse:
    def test_steady_state(self):
        output = np.ones(200)
        t_len, frac = detect_transient_response(output)
        assert isinstance(frac, float)

    def test_short_signal(self):
        t_len, frac = detect_transient_response(np.array([1.0, 2.0]))
        assert t_len == 0

    def test_explicit_transient(self):
        t_len, frac = detect_transient_response(np.ones(100), n_transient_samples=10)
        assert t_len == 10


class TestSignalFilterRules:
    def _get_rules(self):
        return {r.name: r for r in SignalFilterExpansionRuleSet().rules()}

    def test_pole_stability_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_pole_stability_analysis_after_design"], _signal_filter_cdg())
        assert not result.is_failure
        assert "analyze_pole_stability" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_passband_ripple_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_passband_ripple_measurement_after_validate"], _signal_filter_cdg())
        assert not result.is_failure

    def test_group_delay_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_group_delay_analysis_after_frequency_response"], _signal_filter_cdg())
        assert not result.is_failure

    def test_transient_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_transient_detection_after_apply_filter"], _signal_filter_cdg())
        assert not result.is_failure


class TestSignalFilterDiagnostics:
    def test_diagnose_unstable_poles(self):
        diags = SignalFilterExpansionRuleSet().diagnose(_signal_filter_cdg(), ExpansionContext(intermediates={"max_pole_magnitude": 1.05}))
        assert "insert_pole_stability_analysis_after_design" in {d.rule_name for d in diags}

    def test_stable_no_trigger(self):
        diags = SignalFilterExpansionRuleSet().diagnose(_signal_filter_cdg(), ExpansionContext(intermediates={"max_pole_magnitude": 0.9}))
        assert not [d for d in diags if d.rule_name == "insert_pole_stability_analysis_after_design"]

    def test_diagnose_ripple(self):
        diags = SignalFilterExpansionRuleSet().diagnose(_signal_filter_cdg(), ExpansionContext(intermediates={"passband_ripple_db": 2.5}))
        assert "insert_passband_ripple_measurement_after_validate" in {d.rule_name for d in diags}

    def test_diagnose_group_delay(self):
        diags = SignalFilterExpansionRuleSet().diagnose(_signal_filter_cdg(), ExpansionContext(intermediates={"group_delay_variation": 5.0}))
        assert "insert_group_delay_analysis_after_frequency_response" in {d.rule_name for d in diags}

    def test_diagnose_transient(self):
        diags = SignalFilterExpansionRuleSet().diagnose(_signal_filter_cdg(), ExpansionContext(intermediates={"transient_energy_fraction": 0.3}))
        assert "insert_transient_detection_after_apply_filter" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert SignalFilterExpansionRuleSet().diagnose(_signal_filter_cdg(), ExpansionContext()) == []


class TestSignalFilterIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([SignalFilterExpansionRuleSet()]).expand(
            _signal_filter_cdg(), ExpansionContext(intermediates={"max_pole_magnitude": 1.1, "passband_ripple_db": 2.0}))
        assert result.expanded

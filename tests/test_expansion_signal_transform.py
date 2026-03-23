"""Tests for the Signal Transform expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.signal_transform import SignalTransformExpansionRuleSet
from sciona.expansion_atoms.runtime_signal_transform import (
    analyze_window_leakage, detect_spectral_aliasing,
    validate_parseval_energy, check_inverse_reconstruction,
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

def _signal_transform_cdg():
    return _cdg(
        [_node("src", "Source"), _node("w", "Window", ConceptType.SIGNAL_TRANSFORM),
         _node("ft", "Forward Transform", ConceptType.SIGNAL_TRANSFORM),
         _node("sp", "Spectral Processing", ConceptType.SIGNAL_TRANSFORM),
         _node("it", "Inverse Transform", ConceptType.SIGNAL_TRANSFORM),
         _node("out", "Output")],
        [_edge("src", "w"), _edge("w", "ft"), _edge("ft", "sp"), _edge("sp", "it"), _edge("it", "out")],
    )


class TestAnalyzeWindowLeakage:
    def test_no_leakage(self):
        sig = np.ones(100)
        ratio, excessive = analyze_window_leakage(sig, sig)
        assert ratio == 0.0
        assert not excessive

    def test_high_leakage(self):
        original = np.ones(100)
        windowed = np.zeros(100)
        ratio, excessive = analyze_window_leakage(windowed, original)
        assert ratio == 1.0
        assert excessive

    def test_zero_original(self):
        ratio, excessive = analyze_window_leakage(np.zeros(10), np.zeros(10))
        assert ratio == 0.0
        assert not excessive


class TestDetectSpectralAliasing:
    def test_no_aliasing(self):
        spectrum = np.zeros(100, dtype=complex)
        spectrum[5] = 10.0  # low frequency
        frac, aliased = detect_spectral_aliasing(spectrum)
        assert not aliased

    def test_has_aliasing(self):
        spectrum = np.zeros(100, dtype=complex)
        spectrum[95] = 10.0  # near Nyquist
        frac, aliased = detect_spectral_aliasing(spectrum)
        assert aliased

    def test_empty(self):
        frac, aliased = detect_spectral_aliasing(np.array([], dtype=complex))
        assert frac == 0.0
        assert not aliased


class TestValidateParsevalEnergy:
    def test_valid_transform(self):
        t = np.random.RandomState(42).randn(64)
        f = np.fft.fft(t)
        err, valid = validate_parseval_energy(t, f)
        assert valid
        assert err < 1e-6

    def test_invalid_transform(self):
        t = np.ones(64)
        f = np.zeros(64, dtype=complex)
        err, valid = validate_parseval_energy(t, f)
        assert not valid

    def test_zero_signal(self):
        err, valid = validate_parseval_energy(np.zeros(10), np.zeros(10, dtype=complex))
        assert valid


class TestCheckInverseReconstruction:
    def test_perfect_reconstruction(self):
        sig = np.random.RandomState(42).randn(64)
        err, faithful = check_inverse_reconstruction(sig, sig)
        assert faithful
        assert err == 0.0

    def test_lossy_reconstruction(self):
        sig = np.ones(64)
        recon = sig + 0.1
        err, faithful = check_inverse_reconstruction(sig, recon)
        assert not faithful

    def test_zero_signal(self):
        err, faithful = check_inverse_reconstruction(np.zeros(10), np.zeros(10))
        assert faithful


class TestSignalTransformRules:
    def _get_rules(self):
        return {r.name: r for r in SignalTransformExpansionRuleSet().rules()}

    def test_window_leakage_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_window_leakage_analysis_after_window"], _signal_transform_cdg())
        assert not result.is_failure
        assert "analyze_window_leakage" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_aliasing_detection_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_aliasing_detection_after_forward_transform"], _signal_transform_cdg())
        assert not result.is_failure

    def test_parseval_validation_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_parseval_validation_after_spectral_processing"], _signal_transform_cdg())
        assert not result.is_failure

    def test_reconstruction_check_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_reconstruction_check_after_inverse"], _signal_transform_cdg())
        assert not result.is_failure


class TestSignalTransformDiagnostics:
    def test_diagnose_window_leakage(self):
        diags = SignalTransformExpansionRuleSet().diagnose(_signal_transform_cdg(), ExpansionContext(intermediates={"window_leakage_ratio": 0.7}))
        assert "insert_window_leakage_analysis_after_window" in {d.rule_name for d in diags}

    def test_low_leakage_no_trigger(self):
        diags = SignalTransformExpansionRuleSet().diagnose(_signal_transform_cdg(), ExpansionContext(intermediates={"window_leakage_ratio": 0.1}))
        assert not [d for d in diags if d.rule_name == "insert_window_leakage_analysis_after_window"]

    def test_diagnose_aliasing(self):
        diags = SignalTransformExpansionRuleSet().diagnose(_signal_transform_cdg(), ExpansionContext(intermediates={"alias_energy_fraction": 0.3}))
        assert "insert_aliasing_detection_after_forward_transform" in {d.rule_name for d in diags}

    def test_diagnose_parseval(self):
        diags = SignalTransformExpansionRuleSet().diagnose(_signal_transform_cdg(), ExpansionContext(intermediates={"parseval_relative_error": 0.01}))
        assert "insert_parseval_validation_after_spectral_processing" in {d.rule_name for d in diags}

    def test_diagnose_reconstruction(self):
        diags = SignalTransformExpansionRuleSet().diagnose(_signal_transform_cdg(), ExpansionContext(intermediates={"reconstruction_error": 0.001}))
        assert "insert_reconstruction_check_after_inverse" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert SignalTransformExpansionRuleSet().diagnose(_signal_transform_cdg(), ExpansionContext()) == []


class TestSignalTransformIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([SignalTransformExpansionRuleSet()]).expand(
            _signal_transform_cdg(), ExpansionContext(intermediates={"window_leakage_ratio": 0.8, "alias_energy_fraction": 0.2}))
        assert result.expanded

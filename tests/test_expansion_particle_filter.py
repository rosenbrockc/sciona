"""Tests for the Particle Filter expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, DependencyEdge, IOSpec, NodeStatus
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.particle_filter import ParticleFilterExpansionRuleSet
from sciona.expansion_atoms.runtime_particle_filter import (
    monitor_effective_sample_size, analyze_particle_diversity,
    track_weight_variance, check_resampling_quality,
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

def _particle_filter_cdg():
    return _cdg(
        [_node("src", "Source"), _node("pp", "Preprocess", ConceptType.SEQUENTIAL_FILTER),
         _node("pr", "Predict", ConceptType.SEQUENTIAL_FILTER),
         _node("rw", "Reweight", ConceptType.PROBABILISTIC_ORACLE),
         _node("po", "Postprocess", ConceptType.SEQUENTIAL_FILTER),
         _node("out", "Output")],
        [_edge("src", "pp"), _edge("pp", "pr"), _edge("pr", "rw"), _edge("rw", "po"), _edge("po", "out")],
    )


class TestMonitorEffectiveSampleSize:
    def test_healthy(self):
        # Uniform weights
        lw = np.zeros(100)
        ess_frac, healthy = monitor_effective_sample_size(lw)
        assert healthy
        assert ess_frac > 0.99

    def test_degenerate(self):
        # One dominant weight
        lw = np.full(100, -100.0)
        lw[0] = 0.0
        ess_frac, healthy = monitor_effective_sample_size(lw)
        assert not healthy

    def test_empty(self):
        ess_frac, healthy = monitor_effective_sample_size(np.array([]))
        assert not healthy


class TestAnalyzeParticleDiversity:
    def test_diverse(self):
        rng = np.random.RandomState(42)
        particles = rng.randn(100, 2) * 10
        dist, diverse = analyze_particle_diversity(particles)
        assert diverse

    def test_collapsed(self):
        particles = np.ones((100, 2)) * 5.0
        dist, diverse = analyze_particle_diversity(particles)
        assert not diverse

    def test_single(self):
        dist, diverse = analyze_particle_diversity(np.array([[1.0, 2.0]]))
        assert not diverse


class TestTrackWeightVariance:
    def test_stable(self):
        # Constant variance
        history = np.random.RandomState(42).randn(20, 50) * 0.5
        trend, stable = track_weight_variance(history)
        assert isinstance(trend, float)

    def test_single_step(self):
        trend, stable = track_weight_variance(np.array([[1.0, 2.0, 3.0]]))
        assert stable


class TestCheckResamplingQuality:
    def test_good(self):
        # Each particle sampled once
        indices = np.arange(100)
        frac, ok = check_resampling_quality(indices, 100)
        assert ok
        assert frac == 0.01

    def test_aggressive(self):
        # One particle duplicated many times
        indices = np.zeros(100, dtype=int)
        frac, ok = check_resampling_quality(indices, 100)
        assert not ok

    def test_empty(self):
        frac, ok = check_resampling_quality(np.array([], dtype=int), 0)
        assert ok


class TestParticleFilterRules:
    def _get_rules(self):
        return {r.name: r for r in ParticleFilterExpansionRuleSet().rules()}

    def test_ess_monitoring_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_ess_monitoring_after_reweight"], _particle_filter_cdg())
        assert not result.is_failure
        assert "monitor_effective_sample_size" in {n.matched_primitive for n in result.unwrap().nodes if n.matched_primitive}

    def test_diversity_analysis_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_particle_diversity_analysis_after_predict"], _particle_filter_cdg())
        assert not result.is_failure

    def test_weight_variance_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_weight_variance_tracking_after_postprocess"], _particle_filter_cdg())
        assert not result.is_failure

    def test_resampling_quality_applies(self):
        result = GraphRewriter().apply_rule(self._get_rules()["insert_resampling_quality_check_after_preprocess"], _particle_filter_cdg())
        assert not result.is_failure


class TestParticleFilterDiagnostics:
    def test_diagnose_low_ess(self):
        diags = ParticleFilterExpansionRuleSet().diagnose(_particle_filter_cdg(), ExpansionContext(intermediates={"ess_fraction": 0.1}))
        assert "insert_ess_monitoring_after_reweight" in {d.rule_name for d in diags}

    def test_healthy_ess_no_trigger(self):
        diags = ParticleFilterExpansionRuleSet().diagnose(_particle_filter_cdg(), ExpansionContext(intermediates={"ess_fraction": 0.8}))
        assert not [d for d in diags if d.rule_name == "insert_ess_monitoring_after_reweight"]

    def test_diagnose_low_diversity(self):
        diags = ParticleFilterExpansionRuleSet().diagnose(_particle_filter_cdg(), ExpansionContext(intermediates={"particle_diversity_low": True}))
        assert "insert_particle_diversity_analysis_after_predict" in {d.rule_name for d in diags}

    def test_diagnose_weight_variance(self):
        diags = ParticleFilterExpansionRuleSet().diagnose(_particle_filter_cdg(), ExpansionContext(intermediates={"weight_variance_trend": 0.05}))
        assert "insert_weight_variance_tracking_after_postprocess" in {d.rule_name for d in diags}

    def test_diagnose_resampling(self):
        diags = ParticleFilterExpansionRuleSet().diagnose(_particle_filter_cdg(), ExpansionContext(intermediates={"max_duplication_fraction": 0.3}))
        assert "insert_resampling_quality_check_after_preprocess" in {d.rule_name for d in diags}

    def test_no_data_returns_nothing(self):
        assert ParticleFilterExpansionRuleSet().diagnose(_particle_filter_cdg(), ExpansionContext()) == []


class TestParticleFilterIntegration:
    def test_full_expansion(self):
        result = ExpansionEngine([ParticleFilterExpansionRuleSet()]).expand(
            _particle_filter_cdg(), ExpansionContext(intermediates={"ess_fraction": 0.1, "max_duplication_fraction": 0.5}))
        assert result.expanded

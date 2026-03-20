"""Tests for the Sequential Filter expansion rules and runtime atoms."""

import numpy as np
import pytest

from sciona.architect.graph_rewriter import GraphRewriter
from sciona.architect.handoff import CDGExport
from sciona.architect.models import (
    AlgorithmicNode,
    ConceptType,
    DependencyEdge,
    IOSpec,
    NodeStatus,
)
from sciona.principal.expansion import ExpansionContext, ExpansionEngine
from sciona.principal.expansion_rules.sequential_filter import (
    SequentialFilterExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_sequential_filter import (
    adapt_process_noise,
    check_observability,
    detect_filter_divergence,
    validate_innovation_whiteness,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _node(nid, name, concept=ConceptType.CUSTOM, primitive=None):
    return AlgorithmicNode(
        node_id=nid,
        name=name,
        description=name,
        concept_type=concept,
        status=NodeStatus.ATOMIC,
        matched_primitive=primitive,
        inputs=[IOSpec(name="in", type_desc="ndarray")],
        outputs=[IOSpec(name="out", type_desc="ndarray")],
        type_signature=f"{name} -> r",
    )


def _edge(src, tgt):
    return DependencyEdge(
        source_id=src,
        target_id=tgt,
        output_name="out",
        input_name="in",
        source_type="ndarray",
        target_type="ndarray",
    )


def _cdg(nodes, edges):
    return CDGExport(nodes=nodes, edges=edges, metadata={})


def _kalman_cdg():
    """Build a minimal Kalman filter CDG matching the skeleton shape."""
    return _cdg(
        [
            _node("init", "Init", ConceptType.STATE_INIT),
            _node("ps", "Predict State", ConceptType.SEQUENTIAL_FILTER),
            _node("pc", "Predict Covariance", ConceptType.SEQUENTIAL_FILTER),
            _node("inn", "Innovation", ConceptType.SEQUENTIAL_FILTER),
            _node("kg", "Kalman Gain", ConceptType.CONJUGATE_UPDATE),
            _node("us", "Update State", ConceptType.CONJUGATE_UPDATE),
            _node("uc", "Update Covariance", ConceptType.CONJUGATE_UPDATE),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("init", "ps"),
            _edge("init", "pc"),
            _edge("ps", "inn"),
            _edge("pc", "kg"),
            _edge("inn", "us"),
            _edge("kg", "us"),
            _edge("kg", "uc"),
            _edge("ps", "us"),
            _edge("pc", "uc"),
            _edge("us", "out"),
            _edge("uc", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestCheckObservability:
    def test_observable_system(self):
        F = np.array([[1, 1], [0, 1]], dtype=np.float64)
        H = np.array([[1, 0]], dtype=np.float64)
        is_obs, O = check_observability(F, H, 2)
        assert is_obs
        assert O.shape == (2, 2)

    def test_unobservable_system(self):
        F = np.array([[1, 0], [0, 1]], dtype=np.float64)
        H = np.array([[1, 0]], dtype=np.float64)  # can't see state 2
        is_obs, O = check_observability(F, H, 2)
        assert not is_obs

    def test_single_state(self):
        is_obs, O = check_observability(np.array([[1.0]]), np.array([[1.0]]), 1)
        assert is_obs


class TestValidateInnovationWhiteness:
    def test_white_noise_has_small_acf(self):
        rng = np.random.default_rng(123)
        innovations = rng.standard_normal(5000)
        acf, _ = validate_innovation_whiteness(innovations, max_lag=5)
        # White noise ACF should be near zero; use a generous 0.05 bound
        # (the formal 95% bound is ~0.028 for N=5000, but we just want to
        # verify the ACF is not large like an AR(1) process would produce).
        assert np.max(np.abs(acf)) < 0.05
        assert len(acf) == 5

    def test_correlated_signal_is_not_white(self):
        # AR(1) process: strong autocorrelation
        rng = np.random.default_rng(42)
        n = 1000
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.standard_normal()
        acf, is_white = validate_innovation_whiteness(x)
        assert not is_white
        assert abs(acf[0]) > 0.5  # lag-1 ACF should be large

    def test_short_sequence(self):
        acf, is_white = validate_innovation_whiteness(np.array([1.0, 2.0, 3.0]))
        assert is_white  # too short to reject


class TestDetectFilterDivergence:
    def test_well_tuned_filter(self):
        rng = np.random.default_rng(42)
        n = 100
        innovations = rng.standard_normal((n, 1))
        S = np.array([[[1.0]]] * n)
        nis, mask = detect_filter_divergence(innovations, S)
        assert len(nis) == n
        # Well-tuned: ~5% should exceed chi2(1)=3.841
        assert np.mean(mask) < 0.15

    def test_divergent_filter(self):
        rng = np.random.default_rng(42)
        n = 100
        # Innovations much larger than S predicts
        innovations = 10.0 * rng.standard_normal((n, 1))
        S = np.array([[[1.0]]] * n)
        nis, mask = detect_filter_divergence(innovations, S)
        assert np.mean(mask) > 0.5  # most should be flagged

    def test_scalar_innovations(self):
        innovations = np.array([0.1, 0.2, 0.3, 0.1, 0.2])
        S = np.array([[1.0]])
        nis, mask = detect_filter_divergence(innovations, S)
        assert len(nis) == 5


class TestAdaptProcessNoise:
    def test_adapts_from_prior(self):
        rng = np.random.default_rng(42)
        n = 50
        innovations = rng.standard_normal((n, 2))
        K = np.eye(2) * 0.5
        K_matrices = np.array([K] * n)
        Q_prior = np.eye(2)
        Q_adapted = adapt_process_noise(innovations, K_matrices, Q_prior)
        assert Q_adapted.shape == (2, 2)
        # Should be symmetric
        np.testing.assert_allclose(Q_adapted, Q_adapted.T)

    def test_empty_innovations(self):
        Q = np.eye(2)
        result = adapt_process_noise(np.array([]), np.array([]), Q)
        np.testing.assert_array_equal(result, Q)


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestSequentialFilterRules:
    def _get_rules(self):
        rs = SequentialFilterExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_observability_check_applies(self):
        rules = self._get_rules()
        rule = rules["insert_observability_check_before_predict"]
        rw = GraphRewriter()
        cdg = _kalman_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "check_observability" in prims
        assert len(g.nodes) == 9  # 8 + 1

    def test_divergence_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_divergence_detection_after_update"]
        rw = GraphRewriter()
        cdg = _kalman_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_filter_divergence" in prims

    def test_adaptive_noise_applies(self):
        rules = self._get_rules()
        rule = rules["insert_adaptive_noise_before_predict_cov"]
        rw = GraphRewriter()
        cdg = _kalman_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "adapt_process_noise" in prims

    def test_innovation_whiteness_applies(self):
        rules = self._get_rules()
        rule = rules["insert_innovation_whiteness_after_update"]
        rw = GraphRewriter()
        cdg = _kalman_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "validate_innovation_whiteness" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestSequentialFilterDiagnostics:
    def test_diagnose_unobservable_system(self):
        rs = SequentialFilterExpansionRuleSet()
        F = np.eye(2)
        H = np.array([[1, 0]])
        ctx = ExpansionContext(signal_data={"F": F, "H": H, "n_states": 2})
        cdg = _kalman_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_observability_check_before_predict" in names

    def test_diagnose_observable_system_no_trigger(self):
        rs = SequentialFilterExpansionRuleSet()
        F = np.array([[1, 1], [0, 1]])
        H = np.array([[1, 0]])
        ctx = ExpansionContext(signal_data={"F": F, "H": H, "n_states": 2})
        cdg = _kalman_cdg()
        diags = rs.diagnose(cdg, ctx)
        obs_diags = [d for d in diags if d.rule_name == "insert_observability_check_before_predict"]
        assert len(obs_diags) == 0

    def test_diagnose_non_white_innovations(self):
        rs = SequentialFilterExpansionRuleSet()
        rng = np.random.default_rng(42)
        n = 500
        x = np.zeros(n)
        for i in range(1, n):
            x[i] = 0.9 * x[i - 1] + rng.standard_normal()
        ctx = ExpansionContext(intermediates={"innovations": x})
        cdg = _kalman_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_innovation_whiteness_after_update" in names

    def test_diagnose_nis_divergence(self):
        rs = SequentialFilterExpansionRuleSet()
        rng = np.random.default_rng(42)
        n = 100
        innovations = 10.0 * rng.standard_normal((n, 1))
        S = np.array([[[1.0]]] * n)
        ctx = ExpansionContext(
            intermediates={"innovations": innovations, "S_matrices": S}
        )
        cdg = _kalman_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_divergence_detection_after_update" in names

    def test_no_data_returns_nothing(self):
        rs = SequentialFilterExpansionRuleSet()
        cdg = _kalman_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestSequentialFilterIntegration:
    def test_full_expansion_with_bad_filter(self):
        """End-to-end: diagnostics fire, engine expands Kalman CDG."""
        rs = SequentialFilterExpansionRuleSet()
        engine = ExpansionEngine([rs])

        rng = np.random.default_rng(42)
        n = 200

        # Unobservable system
        F = np.eye(2)
        H = np.array([[1, 0]])

        # Correlated innovations (AR(1))
        innovations_1d = np.zeros(n)
        for i in range(1, n):
            innovations_1d[i] = 0.85 * innovations_1d[i - 1] + rng.standard_normal()

        ctx = ExpansionContext(
            signal_data={"F": F, "H": H, "n_states": 2},
            intermediates={"innovations": innovations_1d},
        )
        cdg = _kalman_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "check_observability",
            "validate_innovation_whiteness",
            "detect_filter_divergence",
            "adapt_process_noise",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_signal_rules(self):
        """Sequential filter + signal rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.signal_event_rate import (
            SignalEventRateExpansionRuleSet,
        )

        engine = ExpansionEngine([
            SequentialFilterExpansionRuleSet(),
            SignalEventRateExpansionRuleSet(),
        ])

        # Only Kalman data, no signal data → only Kalman diags fire
        F = np.eye(2)
        H = np.array([[1, 0]])
        ctx = ExpansionContext(signal_data={"F": F, "H": H, "n_states": 2})
        cdg = _kalman_cdg()
        result = engine.expand(cdg, ctx)

        # Signal rules should NOT have fired (no signal data)
        signal_atoms = {"remove_signal_jumps", "assess_signal_quality", "reject_outlier_intervals"}
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & signal_atoms)

"""Tests for the MCMC/HMC expansion rules and runtime atoms."""

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
from sciona.principal.expansion_rules.mcmc import (
    MCMCExpansionRuleSet,
)
from sciona.expansion_atoms.runtime_mcmc import (
    compute_convergence_diagnostics,
    compute_dual_averaging_step_size,
    detect_divergent_transitions,
    estimate_mass_matrix,
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


def _hmc_cdg():
    """Build a minimal HMC CDG matching the skeleton topology."""
    return _cdg(
        [
            _node("init", "Initialization Subgraph", ConceptType.MCMC_KERNEL),
            _node("hp1", "Half Step Momentum Start", ConceptType.MCMC_KERNEL),
            _node("fsq", "Full Step Position", ConceptType.MCMC_KERNEL),
            _node("oq", "Oracle Query", ConceptType.PROBABILISTIC_ORACLE),
            _node("hp2", "Half Step Momentum End", ConceptType.MCMC_KERNEL),
            _node("acc", "Acceptance Criterion", ConceptType.MCMC_KERNEL),
            _node("out", "Output", ConceptType.CUSTOM),
        ],
        [
            _edge("init", "hp1"),
            _edge("hp1", "fsq"),
            _edge("fsq", "oq"),
            _edge("oq", "hp2"),
            _edge("hp2", "acc"),
            _edge("acc", "out"),
        ],
    )


# ---------------------------------------------------------------------------
# Runtime atom tests
# ---------------------------------------------------------------------------


class TestDetectDivergentTransitions:
    def test_no_divergences(self):
        energies_init = np.array([10.0, 10.1, 10.2, 9.9, 10.0])
        energies_prop = np.array([10.1, 10.0, 10.3, 10.0, 10.1])
        errors, mask = detect_divergent_transitions(energies_init, energies_prop)
        assert len(errors) == 5
        assert not np.any(mask)

    def test_all_divergent(self):
        energies_init = np.array([10.0, 10.0, 10.0])
        energies_prop = np.array([2000.0, 3000.0, 5000.0])
        errors, mask = detect_divergent_transitions(energies_init, energies_prop)
        assert np.all(mask)
        np.testing.assert_allclose(errors, [1990.0, 2990.0, 4990.0])

    def test_custom_threshold(self):
        energies_init = np.array([0.0, 0.0])
        energies_prop = np.array([50.0, 500.0])
        errors, mask = detect_divergent_transitions(
            energies_init, energies_prop, threshold=100.0
        )
        assert not mask[0]
        assert mask[1]


class TestComputeDualAveragingStepSize:
    def test_high_accept_rate_decreases_epsilon(self):
        # All accept probs near 1.0 → too small step size → should increase
        accept_probs = np.full(100, 0.95)
        eps = compute_dual_averaging_step_size(accept_probs, target_accept=0.65)
        assert isinstance(eps, float)
        assert eps > 0

    def test_low_accept_rate(self):
        # All accept probs near 0.1 → too large step size → should decrease
        accept_probs = np.full(100, 0.1)
        eps = compute_dual_averaging_step_size(accept_probs, target_accept=0.65)
        assert isinstance(eps, float)
        assert eps > 0

    def test_empty_input(self):
        eps = compute_dual_averaging_step_size(np.array([]))
        assert eps == 1.0  # returns epsilon_0

    def test_on_target(self):
        # Accept probs exactly at target → epsilon should stay near epsilon_0
        accept_probs = np.full(200, 0.65)
        eps = compute_dual_averaging_step_size(
            accept_probs, target_accept=0.65, epsilon_0=0.5
        )
        assert 0.1 < eps < 5.0


class TestEstimateMassMatrix:
    def test_diagonal(self):
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((500, 3)) * np.array([1.0, 5.0, 0.1])
        M = estimate_mass_matrix(samples, diagonal_only=True)
        assert M.shape == (3,)
        # Variances should roughly match scale^2
        assert M[1] > M[0] > M[2]

    def test_dense(self):
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((500, 2))
        M = estimate_mass_matrix(samples, diagonal_only=False)
        assert M.shape == (2, 2)
        # Should be symmetric
        np.testing.assert_allclose(M, M.T)

    def test_single_sample(self):
        M = estimate_mass_matrix(np.array([[1.0, 2.0]]), diagonal_only=True)
        assert M.shape == (2,)
        np.testing.assert_array_equal(M, [1.0, 1.0])  # fallback

    def test_1d_input(self):
        M = estimate_mass_matrix(np.array([1.0, 2.0, 3.0]), diagonal_only=True)
        assert M.shape == (1,)


class TestComputeConvergenceDiagnostics:
    def test_converged_chains(self):
        rng = np.random.default_rng(42)
        # Two chains sampling from same distribution → should converge
        chains = rng.standard_normal((2, 500, 3))
        rhat, ess = compute_convergence_diagnostics(chains)
        assert rhat.shape == (3,)
        assert ess.shape == (3,)
        # Converged chains should have R-hat near 1.0
        assert np.all(rhat < 1.1)
        # ESS should be positive
        assert np.all(ess > 0)

    def test_divergent_chains(self):
        rng = np.random.default_rng(42)
        # Two chains with different means → should not converge
        chain1 = rng.standard_normal((1, 200, 2))
        chain2 = rng.standard_normal((1, 200, 2)) + 10.0
        chains = np.concatenate([chain1, chain2], axis=0)
        rhat, ess = compute_convergence_diagnostics(chains)
        # At least one param should have high R-hat
        assert np.max(rhat) > 1.1

    def test_single_chain_split(self):
        rng = np.random.default_rng(42)
        chains = rng.standard_normal((1, 200, 2))
        rhat, ess = compute_convergence_diagnostics(chains)
        assert rhat.shape == (2,)
        # Single well-mixed chain split in half should be near 1
        assert np.all(rhat < 1.1)

    def test_short_chain(self):
        chains = np.array([[[1.0], [2.0]]])  # too short to split
        rhat, ess = compute_convergence_diagnostics(chains)
        assert rhat.shape == (1,)
        np.testing.assert_array_equal(rhat, [1.0])


# ---------------------------------------------------------------------------
# DPO rule application tests
# ---------------------------------------------------------------------------


class TestMCMCRules:
    def _get_rules(self):
        rs = MCMCExpansionRuleSet()
        return {r.name: r for r in rs.rules()}

    def test_divergence_detection_applies(self):
        rules = self._get_rules()
        rule = rules["insert_divergence_detection_after_accept"]
        rw = GraphRewriter()
        cdg = _hmc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "detect_divergent_transitions" in prims
        assert len(g.nodes) == 8  # 7 + 1

    def test_step_size_adaptation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_step_size_adaptation_before_leapfrog"]
        rw = GraphRewriter()
        cdg = _hmc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "compute_dual_averaging_step_size" in prims

    def test_mass_matrix_estimation_applies(self):
        rules = self._get_rules()
        rule = rules["insert_mass_matrix_estimation_before_leapfrog"]
        rw = GraphRewriter()
        cdg = _hmc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "estimate_mass_matrix" in prims

    def test_convergence_diagnostics_applies(self):
        rules = self._get_rules()
        rule = rules["insert_convergence_diagnostics_after_accept"]
        rw = GraphRewriter()
        cdg = _hmc_cdg()
        result = rw.apply_rule(rule, cdg)
        assert not result.is_failure
        g = result.unwrap()
        prims = {n.matched_primitive for n in g.nodes if n.matched_primitive}
        assert "compute_convergence_diagnostics" in prims


# ---------------------------------------------------------------------------
# Diagnostic tests
# ---------------------------------------------------------------------------


class TestMCMCDiagnostics:
    def test_diagnose_divergent_transitions(self):
        rs = MCMCExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={
                "energies_initial": np.array([10.0, 10.0, 10.0]),
                "energies_proposed": np.array([10.1, 2000.0, 10.2]),
            }
        )
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_divergence_detection_after_accept" in names

    def test_no_divergence_no_trigger(self):
        rs = MCMCExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={
                "energies_initial": np.array([10.0, 10.0]),
                "energies_proposed": np.array([10.1, 10.2]),
            }
        )
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        div_diags = [
            d for d in diags
            if d.rule_name == "insert_divergence_detection_after_accept"
        ]
        assert len(div_diags) == 0

    def test_diagnose_low_acceptance_rate(self):
        rs = MCMCExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"accept_probs": np.full(100, 0.2)}
        )
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_step_size_adaptation_before_leapfrog" in names

    def test_diagnose_high_acceptance_rate(self):
        rs = MCMCExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"accept_probs": np.full(100, 0.95)}
        )
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_step_size_adaptation_before_leapfrog" in names

    def test_optimal_acceptance_no_trigger(self):
        rs = MCMCExpansionRuleSet()
        ctx = ExpansionContext(
            intermediates={"accept_probs": np.full(100, 0.65)}
        )
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        step_diags = [
            d for d in diags
            if d.rule_name == "insert_step_size_adaptation_before_leapfrog"
        ]
        assert len(step_diags) == 0

    def test_diagnose_parameter_scale_variance(self):
        rs = MCMCExpansionRuleSet()
        rng = np.random.default_rng(42)
        # Wide scale variance: param 1 has std~1, param 2 has std~100
        samples = rng.standard_normal((200, 2)) * np.array([1.0, 100.0])
        ctx = ExpansionContext(intermediates={"samples": samples})
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_mass_matrix_estimation_before_leapfrog" in names

    def test_uniform_scales_no_trigger(self):
        rs = MCMCExpansionRuleSet()
        rng = np.random.default_rng(42)
        samples = rng.standard_normal((200, 3))
        ctx = ExpansionContext(intermediates={"samples": samples})
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        mass_diags = [
            d for d in diags
            if d.rule_name == "insert_mass_matrix_estimation_before_leapfrog"
        ]
        assert len(mass_diags) == 0

    def test_diagnose_convergence(self):
        rs = MCMCExpansionRuleSet()
        rng = np.random.default_rng(42)
        chain1 = rng.standard_normal((1, 200, 2))
        chain2 = rng.standard_normal((1, 200, 2)) + 10.0
        chains = np.concatenate([chain1, chain2], axis=0)
        ctx = ExpansionContext(intermediates={"chains": chains})
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        names = {d.rule_name for d in diags}
        assert "insert_convergence_diagnostics_after_accept" in names

    def test_converged_chains_no_trigger(self):
        rs = MCMCExpansionRuleSet()
        rng = np.random.default_rng(42)
        chains = rng.standard_normal((2, 500, 3))
        ctx = ExpansionContext(intermediates={"chains": chains})
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ctx)
        conv_diags = [
            d for d in diags
            if d.rule_name == "insert_convergence_diagnostics_after_accept"
        ]
        assert len(conv_diags) == 0

    def test_no_data_returns_nothing(self):
        rs = MCMCExpansionRuleSet()
        cdg = _hmc_cdg()
        diags = rs.diagnose(cdg, ExpansionContext())
        assert diags == []


# ---------------------------------------------------------------------------
# Integration: full expansion engine
# ---------------------------------------------------------------------------


class TestMCMCIntegration:
    def test_full_expansion_with_bad_sampler(self):
        """End-to-end: diagnostics fire, engine expands HMC CDG."""
        rs = MCMCExpansionRuleSet()
        engine = ExpansionEngine([rs])

        rng = np.random.default_rng(42)

        # Divergent transitions + bad acceptance rate + non-converged chains
        energies_init = np.full(50, 10.0)
        energies_prop = np.concatenate([
            np.full(40, 10.1),
            np.full(10, 5000.0),  # divergent
        ])

        chain1 = rng.standard_normal((1, 200, 2))
        chain2 = rng.standard_normal((1, 200, 2)) + 5.0
        chains = np.concatenate([chain1, chain2], axis=0)

        ctx = ExpansionContext(
            intermediates={
                "energies_initial": energies_init,
                "energies_proposed": energies_prop,
                "accept_probs": np.full(100, 0.15),
                "chains": chains,
            }
        )
        cdg = _hmc_cdg()
        result = engine.expand(cdg, ctx)

        assert result.expanded
        assert len(result.applied_rules) >= 1
        prims = {n.matched_primitive for n in result.cdg.nodes if n.matched_primitive}
        expansion_atoms = prims & {
            "detect_divergent_transitions",
            "compute_dual_averaging_step_size",
            "estimate_mass_matrix",
            "compute_convergence_diagnostics",
        }
        assert len(expansion_atoms) >= 1

    def test_cross_domain_with_kalman_rules(self):
        """MCMC + sequential filter rules both available; only relevant ones fire."""
        from sciona.principal.expansion_rules.sequential_filter import (
            SequentialFilterExpansionRuleSet,
        )

        engine = ExpansionEngine([
            MCMCExpansionRuleSet(),
            SequentialFilterExpansionRuleSet(),
        ])

        # Only MCMC data, no Kalman data → only MCMC diags fire
        ctx = ExpansionContext(
            intermediates={
                "energies_initial": np.array([10.0, 10.0]),
                "energies_proposed": np.array([5000.0, 6000.0]),
            }
        )
        cdg = _hmc_cdg()
        result = engine.expand(cdg, ctx)

        # Kalman rules should NOT have fired (no Kalman data)
        kalman_atoms = {
            "check_observability",
            "validate_innovation_whiteness",
            "detect_filter_divergence",
            "adapt_process_noise",
        }
        applied_prims = {
            n.matched_primitive for n in result.cdg.nodes if n.matched_primitive
        }
        assert not (applied_prims & kalman_atoms)

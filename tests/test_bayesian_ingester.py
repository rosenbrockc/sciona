"""Tests for Bayesian extensions to the Smart Ingester.

Covers: ConceptType enums, StochasticTraceSpec, emitter witness generation
for Bayesian atoms, and chunker prompt content.
"""

from __future__ import annotations


from ageom.architect.models import ConceptType, IOSpec
from ageom.ingester.models import (
    MacroAtomSpec,
    StateModelSpec,
    StochasticTraceSpec,
    ValidatedMacroPlan,
    ProposedMacroPlan,
)
from ageom.ingester.emitter import (
    _BAYESIAN_CONCEPT_TYPES,
    _generate_bayesian_witness,
    generate_ghost_witnesses,
    generate_state_models,
    emit_ingestion_bundle,
)
from ageom.ingester.prompts import SEMANTIC_CHUNK_SYSTEM

# ---------------------------------------------------------------------------
# ConceptType enums
# ---------------------------------------------------------------------------


class TestBayesianConceptTypes:
    def test_sampler_exists(self):
        assert ConceptType.SAMPLER.value == "sampler"

    def test_log_prob_exists(self):
        assert ConceptType.LOG_PROB.value == "log_prob"

    def test_posterior_update_exists(self):
        assert ConceptType.POSTERIOR_UPDATE.value == "posterior_update"

    def test_variational_inference_exists(self):
        assert ConceptType.VARIATIONAL_INFERENCE.value == "variational_inference"

    def test_prior_init_exists(self):
        assert ConceptType.PRIOR_INIT.value == "prior_init"

    def test_bayesian_types_in_set(self):
        for ct in _BAYESIAN_CONCEPT_TYPES:
            assert isinstance(ct, ConceptType)

    def test_existing_types_unchanged(self):
        # Verify we didn't break existing types
        assert ConceptType.SORTING.value == "sorting"
        assert ConceptType.NEURAL_NETWORK.value == "neural_network"
        assert ConceptType.CUSTOM.value == "custom"


# ---------------------------------------------------------------------------
# StochasticTraceSpec
# ---------------------------------------------------------------------------


class TestStochasticTraceSpec:
    def test_defaults(self):
        spec = StochasticTraceSpec()
        assert spec.rng_field == "rng_key"
        assert spec.rng_type == "jax.random.PRNGKey"
        assert spec.trace_field == ""
        assert spec.chain_count == 1
        assert spec.warmup_steps == 0

    def test_mcmc_spec(self):
        spec = StochasticTraceSpec(
            trace_field="mcmc_trace",
            trace_param_dims=(3,),
            chain_count=4,
            warmup_steps=500,
        )
        assert spec.trace_field == "mcmc_trace"
        assert spec.trace_param_dims == (3,)
        assert spec.chain_count == 4
        assert spec.warmup_steps == 500

    def test_state_model_with_stochastic(self):
        spec = StateModelSpec(
            model_name="BayesianState",
            fields=[("mu", "float"), ("sigma", "float")],
            stochastic=StochasticTraceSpec(
                trace_field="chain",
                trace_param_dims=(2,),
            ),
        )
        assert spec.stochastic is not None
        assert spec.stochastic.trace_field == "chain"

    def test_state_model_without_stochastic(self):
        spec = StateModelSpec(
            model_name="PlainState",
            fields=[("x", "float")],
        )
        assert spec.stochastic is None


# ---------------------------------------------------------------------------
# Chunker prompts
# ---------------------------------------------------------------------------


class TestChunkerPrompts:
    def test_prompt_contains_bayesian_types(self):
        for keyword in [
            "sampler",
            "log_prob",
            "posterior_update",
            "variational_inference",
            "prior_init",
        ]:
            assert keyword in SEMANTIC_CHUNK_SYSTEM

    def test_prompt_contains_bayesian_patterns(self):
        assert "Metropolis-Hastings" in SEMANTIC_CHUNK_SYSTEM
        assert "log_likelihood" in SEMANTIC_CHUNK_SYSTEM
        assert "conjugate" in SEMANTIC_CHUNK_SYSTEM
        assert "ELBO" in SEMANTIC_CHUNK_SYSTEM
        assert "reparameterization" in SEMANTIC_CHUNK_SYSTEM

    def test_prompt_contains_prior_patterns(self):
        assert "prior" in SEMANTIC_CHUNK_SYSTEM.lower()
        assert "hyperparameter" in SEMANTIC_CHUNK_SYSTEM.lower()


# ---------------------------------------------------------------------------
# Bayesian witness generation
# ---------------------------------------------------------------------------


def _make_atom(name: str, concept_type: ConceptType) -> MacroAtomSpec:
    return MacroAtomSpec(
        name=name,
        description=f"Test {name}",
        concept_type=concept_type,
        inputs=[IOSpec(name="x", type_desc="ndarray")],
        outputs=[IOSpec(name="result", type_desc="ndarray")],
    )


class TestBayesianWitnessGeneration:
    def test_prior_init_witness(self):
        atom = _make_atom("Init Prior", ConceptType.PRIOR_INIT)
        lines = _generate_bayesian_witness(
            atom, "init_prior", "witness_init_prior", False
        )
        code = "\n".join(lines)
        assert "AbstractDistribution" in code
        assert "event_shape" in code
        assert "def witness_init_prior" in code

    def test_log_prob_witness(self):
        atom = _make_atom("Evaluate Log Prob", ConceptType.LOG_PROB)
        lines = _generate_bayesian_witness(
            atom, "evaluate_log_prob", "witness_evaluate_log_prob", False
        )
        code = "\n".join(lines)
        assert "AbstractScalar" in code
        assert "dist" in code
        assert "samples" in code

    def test_sampler_witness(self):
        atom = _make_atom("MCMC Step", ConceptType.SAMPLER)
        lines = _generate_bayesian_witness(
            atom, "mcmc_step", "witness_mcmc_step", False
        )
        code = "\n".join(lines)
        assert "AbstractMCMCTrace" in code
        assert "AbstractRNGState" in code
        assert "param_dims" in code

    def test_posterior_update_witness(self):
        atom = _make_atom("Update Posterior", ConceptType.POSTERIOR_UPDATE)
        lines = _generate_bayesian_witness(
            atom, "update_posterior", "witness_update_posterior", False
        )
        code = "\n".join(lines)
        assert "assert_conjugate_to" in code
        assert "prior" in code

    def test_vi_elbo_witness(self):
        atom = _make_atom("Compute ELBO", ConceptType.VARIATIONAL_INFERENCE)
        lines = _generate_bayesian_witness(
            atom, "compute_elbo", "witness_compute_elbo", False
        )
        code = "\n".join(lines)
        assert "q_dist" in code
        assert "p_dist" in code
        assert "AbstractScalar" in code


class TestGenerateGhostWitnesses:
    def test_bayesian_imports_added(self):
        atoms = [_make_atom("MCMC Step", ConceptType.SAMPLER)]
        source, names = generate_ghost_witnesses(atoms)
        assert "AbstractDistribution" in source
        assert "AbstractMCMCTrace" in source
        assert "AbstractRNGState" in source

    def test_no_bayesian_imports_for_plain(self):
        atoms = [_make_atom("Filter Signal", ConceptType.SIGNAL_FILTER)]
        source, names = generate_ghost_witnesses(atoms)
        assert "AbstractDistribution" not in source

    def test_mixed_atoms(self):
        atoms = [
            _make_atom("Filter Signal", ConceptType.SIGNAL_FILTER),
            _make_atom("MCMC Step", ConceptType.SAMPLER),
            _make_atom("Compute ELBO", ConceptType.VARIATIONAL_INFERENCE),
        ]
        source, names = generate_ghost_witnesses(atoms)
        # All three should have witnesses
        assert "witness_filter_signal" in source
        assert "witness_mcmc_step" in source
        assert "witness_compute_elbo" in source
        # Bayesian imports present
        assert "AbstractDistribution" in source
        # Name mapping correct
        assert names["Filter Signal"] == "witness_filter_signal"
        assert names["MCMC Step"] == "witness_mcmc_step"

    def test_opaque_atoms_skipped(self):
        atom = MacroAtomSpec(
            name="DL Module",
            concept_type=ConceptType.NEURAL_NETWORK,
            is_opaque=True,
        )
        source, names = generate_ghost_witnesses([atom])
        assert "DL Module" not in names


# ---------------------------------------------------------------------------
# State model generation with stochastic spec
# ---------------------------------------------------------------------------


class TestGenerateStateModelsStochastic:
    def test_plain_state_model(self):
        spec = StateModelSpec(
            model_name="PlainState",
            fields=[("x", "float")],
        )
        source = generate_state_models([spec])
        assert "class PlainState" in source
        assert "rng_key" not in source

    def test_stochastic_state_model(self):
        spec = StateModelSpec(
            model_name="MCMCState",
            fields=[("mu", "float"), ("sigma", "float")],
            stochastic=StochasticTraceSpec(
                rng_field="rng_key",
                rng_type="jax.random.PRNGKey",
                trace_field="mcmc_trace",
                trace_param_dims=(3,),
                chain_count=4,
                warmup_steps=500,
            ),
        )
        source = generate_state_models([spec])
        assert "class MCMCState" in source
        assert "rng_key" in source
        assert "mcmc_trace" in source
        assert "mcmc_step_count" in source
        assert "mcmc_accept_rate" in source
        assert "param_dims=(3,)" in source
        assert "chains=4" in source
        assert "warmup=500" in source
        assert "import numpy as np" in source

    def test_stochastic_no_trace(self):
        """StochasticTraceSpec without trace_field (RNG-only, e.g. VI)."""
        spec = StateModelSpec(
            model_name="VIState",
            fields=[("q_mu", "ndarray")],
            stochastic=StochasticTraceSpec(
                rng_field="rng_state",
                trace_field="",  # no MCMC trace
            ),
        )
        source = generate_state_models([spec])
        assert "rng_state" in source
        assert "mcmc_trace" not in source


# ---------------------------------------------------------------------------
# End-to-end: emit_ingestion_bundle with Bayesian atoms
# ---------------------------------------------------------------------------


class TestEmitBayesianBundle:
    def test_bayesian_bundle_has_witnesses(self):
        atoms = [
            MacroAtomSpec(
                name="Init Prior",
                description="Initialize the prior",
                concept_type=ConceptType.PRIOR_INIT,
                method_names=["init_prior"],
                inputs=[IOSpec(name="dim", type_desc="int")],
                outputs=[IOSpec(name="prior", type_desc="Distribution")],
            ),
            MacroAtomSpec(
                name="MCMC Step",
                description="Run one MCMC step",
                concept_type=ConceptType.SAMPLER,
                method_names=["step"],
                inputs=[IOSpec(name="state", type_desc="ndarray")],
                outputs=[IOSpec(name="new_state", type_desc="ndarray")],
            ),
        ]
        plan = ValidatedMacroPlan(
            plan=ProposedMacroPlan(macro_atoms=atoms),
            all_attrs_accounted=True,
        )
        bundle = emit_ingestion_bundle(plan, "BayesianSampler")

        # Witnesses should reference Bayesian abstract types
        assert "AbstractDistribution" in bundle.generated_witnesses
        assert "AbstractMCMCTrace" in bundle.generated_witnesses

        # CDG should have the atoms
        node_ids = [n.node_id for n in bundle.cdg.nodes]
        assert "init_prior" in node_ids
        assert "mcmc_step" in node_ids

        # Match results should exist
        assert len(bundle.match_results) == 2

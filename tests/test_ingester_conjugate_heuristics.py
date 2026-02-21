"""Tests for conjugate-update heuristic pass and emitter behavior."""

from __future__ import annotations

from ageom.architect.models import ConceptType, IOSpec
from ageom.ingester.emitter import emit_ingestion_bundle
from ageom.ingester.graph import apply_conjugate_heuristics
from ageom.ingester.models import (
    MacroAtomSpec,
    MethodFact,
    ProposedMacroPlan,
    RawDataFlowGraph,
    ValidatedMacroPlan,
)


def _make_conjugate_dfg() -> RawDataFlowGraph:
    return RawDataFlowGraph(
        class_name="BetaBernoulliModel",
        methods=[
            MethodFact(
                name="ingest_data",
                params=["observations"],
                writes=["observations"],
                source_code="self.observations = observations",
            ),
            MethodFact(
                name="accumulate_stats",
                params=["observations"],
                reads=["observations"],
                writes=["successes", "failures"],
                source_code=(
                    "successes = sum(observations)\n"
                    "failures = len(observations) - successes"
                ),
            ),
            MethodFact(
                name="posterior",
                params=[],
                reads=["successes", "failures"],
                writes=["alpha_post", "beta_post"],
                source_code=(
                    "alpha_post = self.alpha + successes\n"
                    "beta_post = self.beta + failures\n"
                    "return Beta(alpha_post, beta_post)"
                ),
            ),
            MethodFact(
                name="construct_distribution",
                params=["alpha_post", "beta_post"],
                reads=["alpha_post", "beta_post"],
                writes=["posterior_dist"],
                source_code="self.posterior_dist = Beta(alpha_post, beta_post)",
            ),
        ],
    )


def _make_initial_plan() -> ValidatedMacroPlan:
    atoms = [
        MacroAtomSpec(
            name="Data Ingestion",
            method_names=["ingest_data"],
            outputs=[IOSpec(name="observations", type_desc="ndarray")],
            concept_type=ConceptType.CUSTOM,
        ),
        MacroAtomSpec(
            name="Posterior Fit",
            method_names=["accumulate_stats", "posterior"],
            inputs=[IOSpec(name="observations", type_desc="ndarray")],
            outputs=[IOSpec(name="posterior_params", type_desc="tuple[float, float]")],
            concept_type=ConceptType.CUSTOM,
        ),
        MacroAtomSpec(
            name="Distribution Construction",
            method_names=["construct_distribution"],
            inputs=[IOSpec(name="posterior_params", type_desc="tuple[float, float]")],
            outputs=[IOSpec(name="posterior_dist", type_desc="Distribution")],
            concept_type=ConceptType.PRIOR_DISTRIBUTION,
        ),
    ]
    return ValidatedMacroPlan(
        plan=ProposedMacroPlan(macro_atoms=atoms, edge_definitions=[]),
        all_attrs_accounted=True,
    )


class TestConjugateHeuristicPass:
    def test_tags_fit_posterior_atom_as_conjugate_update(self):
        dfg = _make_conjugate_dfg()
        plan = _make_initial_plan()

        updated = apply_conjugate_heuristics(dfg, plan)
        posterior_fit = next(
            a for a in updated.plan.macro_atoms if a.name == "Posterior Fit"
        )

        assert posterior_fit.concept_type == ConceptType.CONJUGATE_UPDATE

    def test_adds_linear_data_update_distribution_edges(self):
        dfg = _make_conjugate_dfg()
        plan = _make_initial_plan()

        updated = apply_conjugate_heuristics(dfg, plan)
        edge_set = {(e.source_id, e.target_id) for e in updated.plan.edge_definitions}

        assert ("data_ingestion", "posterior_fit") in edge_set
        assert ("posterior_fit", "distribution_construction") in edge_set


class TestConjugateEmitterBehavior:
    def test_conjugate_witness_bypasses_sampler_logic(self):
        plan = _make_initial_plan()
        dfg = _make_conjugate_dfg()
        updated = apply_conjugate_heuristics(dfg, plan)

        bundle = emit_ingestion_bundle(updated, "BetaBernoulliModel")

        assert "witness_posterior_fit" in bundle.generated_witnesses
        assert "closed-form conjugate update" in bundle.generated_witnesses
        assert "rng.advance" not in bundle.generated_witnesses
        assert "AbstractMCMCTrace" not in bundle.generated_witnesses

    def test_emitter_keeps_linear_conjugate_sequence(self):
        plan = _make_initial_plan()
        dfg = _make_conjugate_dfg()
        updated = apply_conjugate_heuristics(dfg, plan)

        bundle = emit_ingestion_bundle(updated, "BetaBernoulliModel")
        edge_set = {(e.source_id, e.target_id) for e in bundle.cdg.edges}

        assert ("data_ingestion", "posterior_fit") in edge_set
        assert ("posterior_fit", "distribution_construction") in edge_set

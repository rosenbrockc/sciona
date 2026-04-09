from __future__ import annotations

from types import SimpleNamespace

import pytest

from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, ConceptType, IOSpec, NodeStatus
from sciona.architect.planning_contract import build_planning_artifact
from sciona.principal.models import OptimizationMetric
from sciona.principal.heuristic_proposal_policy import (
    HeuristicProposalGuidance,
    build_heuristic_proposal_guidance,
    candidate_action_classes,
)
from sciona.principal.proposal_helpers import (
    ProposalCandidate,
    apply_heuristic_guidance,
    build_expansion_context,
    evaluate_proposal_candidate,
    proposal_structural_delta,
    select_best_proposal,
    summarize_expansion_context,
)
from sciona.heuristics import HeuristicActionClass


@pytest.mark.asyncio
async def test_evaluate_proposal_candidate_returns_infinite_loss_on_synthesis_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cdg = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="leaf",
                name="Leaf",
                description="Atomic leaf",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
            )
        ],
        edges=[],
        metadata={},
    )
    state = SimpleNamespace(
        metric=OptimizationMetric.PRECISION,
        dataset_path="dataset.yml",
    )

    async def _boom(_cdg: CDGExport, _match_results: list[object]) -> object:
        raise RuntimeError("compile failed")

    monkeypatch.setattr(
        "sciona.principal.proposal_helpers.run_ghost_simulation",
        lambda *_args, **_kwargs: SimpleNamespace(),
    )

    deps = SimpleNamespace(
        match_results_fn=lambda _cdg: [],
        synthesize_fn=_boom,
        sandbox=None,
        dataset_varset=None,
        evaluation_spec=None,
    )

    loss, bundle, benchmark, match_results, ghost_report = await evaluate_proposal_candidate(
        state,
        deps,
        cdg,
    )

    assert loss == float("inf")
    assert bundle is None
    assert benchmark is None
    assert match_results == []
    assert ghost_report is not None


def test_build_expansion_context_carries_planning_artifact() -> None:
    planning_artifact = build_planning_artifact(
        goal="Detect ECG peaks",
        thread_id="thread-1",
        paradigm="signal_detect_measure",
        variant_hint="peak_detection",
        root_inputs=[IOSpec(name="signal", type_desc="np.ndarray")],
        root_outputs=[IOSpec(name="events", type_desc="np.ndarray")],
    ).model_dump(mode="json")
    state = SimpleNamespace(
        benchmark=SimpleNamespace(
            runtime_artifacts={
                "stdout_payload": {"global_loss": 1.0},
                "intermediates": {"events": [1, 2, 3]},
                "runtime_inputs": {"features": [0.1, 0.2]},
                "signal_data": {"signal": [0.1, 0.2]},
                "runtime_evidence": {"core": {"input_count": 1}},
            }
        ),
        planning_artifact=planning_artifact,
    )

    ctx = build_expansion_context(state)

    assert ctx.planning_artifact is not None
    assert ctx.planning_artifact["artifact_version"] == "phase1.v1"
    assert ctx.runtime_inputs == {"features": [0.1, 0.2]}
    assert ctx.runtime_evidence == {"core": {"input_count": 1}}
    summary = summarize_expansion_context(ctx)
    assert summary["runtime_input_keys"] == ["features"]
    assert summary["runtime_evidence_keys"] == ["core"]
    assert summary["planning_artifact"]["paradigm"] == "signal_detect_measure"


def test_build_expansion_context_recovers_canonical_keys_from_runtime_evidence() -> None:
    state = SimpleNamespace(
        benchmark=SimpleNamespace(
            runtime_artifacts={
                "runtime_evidence": {
                    "canonical_runtime_context": {
                        "canonical_inputs": {
                            "signal": {
                                "raw_key": "h10_ecg_value",
                            },
                            "sampling_rate": {
                                "raw_key": "ecg_sampling_rate",
                            },
                        }
                    },
                    "telemetry_summary": {
                        "signal": {
                            "discontinuity_count": 12.0,
                        },
                        "intermediates": {
                            "events": {
                                "count": 438.0,
                            }
                        },
                    },
                }
            }
        ),
        planning_artifact={"family_hint": "signal_detect_measure"},
    )

    ctx = build_expansion_context(state)

    assert ctx.runtime_inputs is not None
    assert "signal" in ctx.runtime_inputs
    assert "sampling_rate" in ctx.runtime_inputs
    assert ctx.intermediates == {"events": {"count": 438.0}}


def test_select_best_proposal_prefers_admissible_improvement() -> None:
    baseline_loss = 10.0
    rejected = ProposalCandidate(
        label="expansion",
        candidate_type="semantic_enrichment",
        cdg=CDGExport(nodes=[], edges=[], metadata={}),
        loss=8.0,
        admissibility={"hard_rejected": True, "routed_to_refinement": False},
    )
    accepted = ProposalCandidate(
        label="local_mutation",
        candidate_type="local_mutation",
        cdg=CDGExport(nodes=[], edges=[], metadata={}),
        loss=9.0,
        admissibility={"hard_rejected": False, "routed_to_refinement": False},
    )

    selected = select_best_proposal(
        [rejected, accepted],
        baseline_loss=baseline_loss,
    )

    assert selected is accepted
    assert selected.selection_disposition == "selected"
    assert selected.selection_reason == "best_admissible_improvement"
    assert rejected.selection_disposition == "rejected"
    assert rejected.selection_reason == "proposal_hard_rejected"


def test_select_best_proposal_prefers_candidate_matching_heuristic_guidance() -> None:
    guidance = HeuristicProposalGuidance(
        family="signal_event_rate",
        heuristic_ids=["interval_instability"],
        preferred_action_classes=[HeuristicActionClass.INSERT_CORRECTION],
        registry_asset_id="family.signal_event_rate.heuristics.v1",
    )
    expansion = ProposalCandidate(
        label="expansion",
        candidate_type="semantic_enrichment",
        cdg=CDGExport(nodes=[], edges=[], metadata={}),
        loss=8.0,
        family="signal_event_rate",
        applied_assets=[{"action_classes": ["insert_correction"]}],
        admissibility={"hard_rejected": False, "routed_to_refinement": False},
    )
    mutation = ProposalCandidate(
        label="local_mutation",
        candidate_type="local_mutation",
        cdg=CDGExport(nodes=[], edges=[], metadata={}),
        loss=8.0,
        family="signal_event_rate",
        admissibility={"hard_rejected": False, "routed_to_refinement": False},
    )
    apply_heuristic_guidance(expansion, guidance=guidance)
    apply_heuristic_guidance(mutation, guidance=guidance)

    selected = select_best_proposal([mutation, expansion], baseline_loss=10.0)

    assert selected is expansion
    assert "matches_heuristic_guidance" in selected.selected_reason_codes


def test_candidate_action_classes_resolve_expansion_asset_family_alias() -> None:
    actions = candidate_action_classes(
        "semantic_enrichment",
        family="signal_detect_measure",
        rules_applied=["insert_outlier_rejection_after_detection"],
    )

    assert HeuristicActionClass.INSERT_CORRECTION in actions
    assert HeuristicActionClass.GATE_OR_VALIDATE in actions


def test_select_best_proposal_keeps_refinement_routed_improvement_eligible() -> None:
    baseline = CDGExport(nodes=[], edges=[], metadata={})
    improved = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="leaf",
                name="Leaf",
                description="Improved leaf",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
            )
        ],
        edges=[],
        metadata={},
    )
    candidate = ProposalCandidate(
        label="expansion",
        candidate_type="semantic_enrichment",
        cdg=improved,
        loss=8.0,
        structural_delta=proposal_structural_delta(baseline, improved),
        admissibility={
            "hard_rejected": False,
            "routed_to_refinement": True,
            "decision_count": 1,
        },
    )

    selected = select_best_proposal([candidate], baseline_loss=10.0)

    assert selected is candidate
    assert selected.selection_disposition == "selected"
    assert "improves_baseline" in selected.selected_reason_codes


def test_history_row_emits_typed_and_compatibility_fields() -> None:
    baseline = CDGExport(nodes=[], edges=[], metadata={})
    improved = CDGExport(
        nodes=[
            AlgorithmicNode(
                node_id="leaf",
                name="Leaf",
                description="Improved leaf",
                concept_type=ConceptType.CUSTOM,
                status=NodeStatus.ATOMIC,
            )
        ],
        edges=[],
        metadata={},
    )
    candidate = ProposalCandidate(
        label="local_mutation",
        candidate_type="local_mutation",
        cdg=improved,
        loss=3.0,
        variant_name="fast_path",
        family="graph_optimization",
        structural_delta=proposal_structural_delta(baseline, improved),
    )

    row = candidate.history_row(baseline_loss=5.0)

    assert row["proposal_type"] == "local_mutation"
    assert row["candidate_type"] == "local_mutation"
    assert row["metadata"]["variant_name"] == "fast_path"
    assert row["family"] == "graph_optimization"
    assert row["structural_delta"]["node_count_delta"] == 1


def test_build_heuristic_proposal_guidance_uses_family_registry() -> None:
    guidance = build_heuristic_proposal_guidance(
        planning_artifact={"family_hint": "divide_and_conquer"},
        runtime_artifacts={
            "heuristics": [
                {
                    "heuristic": {
                        "heuristic_id": "coverage_fragmentation",
                    }
                }
            ]
        },
    )

    assert guidance.family == "divide_and_conquer"
    assert guidance.registry_asset_id == "family.divide_and_conquer.heuristics.v1"
    assert guidance.preferred_action_classes[0] == HeuristicActionClass.GATE_OR_VALIDATE


def test_build_heuristic_proposal_guidance_uses_positive_outcome_memory_cautiously() -> None:
    search_trace = [
        {
            "proposal_selection": {
                "baseline_loss": 10.0,
                "selected": "expansion",
                "candidates": [
                    {
                        "label": "expansion",
                        "family": "signal_event_rate",
                        "loss": 8.0,
                        "evidence": {
                            "heuristic_ids": ["interval_instability"],
                            "candidate_action_classes": ["insert_correction"],
                        },
                    }
                ],
            }
        },
        {
            "proposal_selection": {
                "baseline_loss": 11.0,
                "selected": "expansion",
                "candidates": [
                    {
                        "label": "expansion",
                        "family": "signal_event_rate",
                        "loss": 9.0,
                        "evidence": {
                            "heuristic_ids": ["interval_instability"],
                            "candidate_action_classes": ["insert_correction"],
                        },
                    }
                ],
            }
        },
    ]
    guidance = build_heuristic_proposal_guidance(
        planning_artifact={"family_hint": "signal_event_rate"},
        runtime_artifacts={
            "heuristics": [
                {"heuristic": {"heuristic_id": "interval_instability"}},
            ]
        },
        search_trace=search_trace,
    )

    assert guidance.preferred_action_classes[0] == HeuristicActionClass.INSERT_CORRECTION
    assert any(note.startswith("outcome_memory:") for note in guidance.notes)


def test_build_heuristic_proposal_guidance_weights_recurrent_cohort_heuristics() -> None:
    guidance = build_heuristic_proposal_guidance(
        planning_artifact={"family_hint": "signal_event_rate"},
        runtime_artifacts={
            "heuristic_cohort": {
                "cohort_size": 5,
                "evaluated_member_count": 5,
                "heuristics": {
                    "interval_instability": {
                        "occurrence_count": 5,
                        "member_count": 5,
                        "coverage_fraction": 1.0,
                        "mean_confidence": 0.7,
                        "max_confidence": 0.8,
                    },
                    "quality_instability": {
                        "occurrence_count": 1,
                        "member_count": 1,
                        "coverage_fraction": 0.2,
                        "mean_confidence": 0.95,
                        "max_confidence": 0.95,
                    },
                },
            }
        },
    )

    assert guidance.cohort_size == 5
    assert guidance.heuristic_summary["interval_instability"]["member_count"] == 5
    assert guidance.preferred_action_classes[0] == HeuristicActionClass.INSERT_CORRECTION
    assert any(note.startswith("cohort:interval_instability:5/5") for note in guidance.notes)

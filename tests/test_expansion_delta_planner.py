from __future__ import annotations

from sciona.principal.expansion_assets import clear_local_expansion_asset_caches
from sciona.principal.expansion_delta_planner import (
    DeltaAdaptationKind,
    DeltaPlanningQuery,
    plan_expansion_delta,
)


def test_delta_planner_selects_direct_use_when_base_already_covers_solution() -> None:
    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("signal_event_rate",),
            matched_techniques=("filter signal", "detect events", "measure rate"),
            missing_techniques=(),
            base_coverage=1.0,
        )
    )

    assert plan.decision == DeltaAdaptationKind.DIRECT_USE
    assert plan.selected.operation_sequence is None
    assert plan.selected.path == ("base_cdg", "direct_use")
    assert not plan.should_compose_novel


def test_delta_planner_selects_expansion_pack_for_multi_operation_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("ml_model_selection",),
            matched_techniques=("fit estimator", "score validation split"),
            missing_techniques=(
                "k-fold cross validated ensemble",
                "stacking meta learner",
            ),
            stage_names=("model_training", "prediction_ensemble"),
            base_coverage=0.50,
            max_operations_per_sequence=2,
        )
    )

    assert plan.decision == DeltaAdaptationKind.EXPANSION_PACK
    assert plan.selected.projected_coverage == 1.0
    assert plan.selected.missing_terms_after_plan == ()
    assert plan.selected.operation_rule_names[:2] == (
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
    )
    assert plan.selected.path == (
        "base_cdg",
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
        "adapted_cdg",
    )


def test_delta_planner_selects_refinement_for_single_low_intrusion_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("ode_solver",),
            matched_techniques=("evaluate derivative", "advance state", "adapt step size"),
            missing_techniques=("detect stiffness before advancing state",),
            stage_names=("evaluate_derivative", "advance_state"),
            intermediate_keys=("stiffness_ratio",),
            base_coverage=0.75,
            max_operations_per_sequence=1,
        )
    )

    assert plan.decision == DeltaAdaptationKind.REFINEMENT
    assert plan.selected.operation_rule_names == ("insert_stiffness_detection_before_advance",)
    assert plan.selected.intrusion_cost <= 0.20
    assert plan.selected.missing_terms_after_plan == ()


def test_delta_planner_selects_single_expansion_for_insert_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("signal_event_rate",),
            matched_techniques=("filter signal", "detect events", "measure rate"),
            missing_techniques=("remove jumps before filtering",),
            runtime_keys=("signal", "sampling_rate"),
            intermediate_keys=("events",),
            base_coverage=0.75,
            max_operations_per_sequence=1,
        )
    )

    assert plan.decision == DeltaAdaptationKind.EXPANSION
    assert plan.selected.operation_rule_names == ("insert_jump_removal_before_filter",)
    assert plan.selected.projected_coverage == 1.0


def test_delta_planner_selects_true_novel_when_no_operation_covers_gap() -> None:
    clear_local_expansion_asset_caches()

    plan = plan_expansion_delta(
        DeltaPlanningQuery(
            families=("ml_model_selection",),
            matched_techniques=("fit estimator",),
            missing_techniques=("quantum lattice annealing with orbital mechanics",),
            base_coverage=0.50,
        )
    )

    assert plan.decision == DeltaAdaptationKind.TRUE_NOVEL
    assert plan.should_compose_novel
    assert plan.selected.operation_sequence is None
    assert plan.selected.missing_terms_after_plan == (
        "quantum lattice annealing with orbital mechanics",
    )


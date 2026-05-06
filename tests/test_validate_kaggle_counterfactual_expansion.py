from __future__ import annotations

import importlib.util
from pathlib import Path

from sciona.principal.expansion_assets import clear_local_expansion_asset_caches
from sciona.principal.trick_retrieval import SolutionTrick, SolutionTrickRetriever


def _validation_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "validate_kaggle_batch.py"
    spec = importlib.util.spec_from_file_location("validate_kaggle_batch", path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ml_template() -> dict:
    return {
        "family": "ml_model_selection",
        "paradigm": "supervised_learning",
        "inputs": [{"name": "X_train"}, {"name": "y_train"}],
        "outputs": [{"name": "predictions"}],
        "stages": [
            {
                "stage_id": "model_training",
                "name": "Model training",
                "description": "fit base estimator",
                "concept_type": "model_training",
                "inputs": [{"name": "features"}],
                "outputs": [{"name": "model"}],
            },
            {
                "stage_id": "prediction_ensemble",
                "name": "Prediction ensemble",
                "description": "combine predictions",
                "concept_type": "analysis",
                "inputs": [{"name": "model"}],
                "outputs": [{"name": "predictions"}],
            },
        ],
    }


def test_counterfactual_expansion_reports_base_and_adapted_coverage() -> None:
    clear_local_expansion_asset_caches()
    module = _validation_module()

    base_evaluation = {
        "technique_coverage": 0.333,
        "grounding_rate": 0.80,
        "covered_techniques": ["fit estimator"],
        "missing_techniques": [
            "k-fold cross validated ensemble",
            "stacking meta learner",
        ],
        "coverage_source": "keyword_heuristic",
    }

    adapted, delta_plan = module.evaluate_counterfactual_expansion(
        template_match={"template": "ml_model_selection_template", "family": "ml_model_selection"},
        template=_ml_template(),
        base_evaluation=base_evaluation,
        key_techniques=[
            "fit estimator",
            "k-fold cross validated ensemble",
            "stacking meta learner",
        ],
        max_rounds=2,
    )

    assert adapted["technique_coverage"] == 1.0
    assert adapted["grounding_rate"] == 0.80
    assert adapted["missing_techniques"] == []
    assert adapted["counterfactual_expansion"]["decision"] == "expansion_pack"
    assert delta_plan["operation_rule_names"] == [
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
    ]


def test_counterfactual_expansion_marks_rescued_competitions() -> None:
    module = _validation_module()

    assert module.is_rescued_by_expansion(
        "divergent",
        "competitive",
        {"operation_rule_names": ["apply_kfold_ensemble"]},
    )
    assert not module.is_rescued_by_expansion(
        "partial",
        "partial",
        {"operation_rule_names": ["apply_kfold_ensemble"]},
    )
    assert not module.is_rescued_by_expansion(
        "divergent",
        "competitive",
        {"operation_rule_names": []},
    )


def test_build_trick_telemetry_reports_available_and_suppressed_tricks() -> None:
    module = _validation_module()
    retriever = SolutionTrickRetriever(
        [
            SolutionTrick(
                trick_id="trick.test.metric_bound_clipping",
                name="Metric-bound clipping",
                kind="metric_hack",
                status="allowed_with_validation",
                risk_level="medium",
                generalization_level="general",
                summary="Clip regression predictions to metric bounds.",
                applies_when=("metric bounds define valid prediction ranges",),
                validation_requirements=("held-out ablation",),
                related_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
                tags=("metric", "clipping"),
            ),
            SolutionTrick(
                trick_id="trick.test.public_lb_probe",
                name="Public leaderboard probing",
                kind="public_lb_overfit_risk",
                status="cataloged",
                risk_level="high",
                generalization_level="competition_specific",
                summary="Tune thresholds against public leaderboard feedback.",
                related_cdgs=("solution.kaggle.classical_tabular_ensemble_topology",),
                tags=("leaderboard", "thresholding"),
            ),
        ]
    )

    telemetry = module.build_trick_telemetry(
        prompt="tabular regression with RMSLE metric clipping",
        title="Regression Challenge",
        solution_summary="Winning solution clipped metric-bounded predictions.",
        matches=[
            {
                "template": "solution.kaggle.classical_tabular_ensemble_topology",
                "family": "ml_model_selection",
                "paradigm": "supervised_learning",
            }
        ],
        assessment="divergent",
        base_assessment="divergent",
        evaluation={
            "technique_coverage": 0.25,
            "coverage_source": "llm_semantic",
            "missing_techniques": ["metric-bound clipping"],
        },
        base_evaluation=None,
        delta_plan={"should_compose_novel": True, "projected_coverage": 0.25},
        trick_retriever=retriever,
    )

    assert telemetry["novel_cdg_required"] is True
    assert telemetry["candidate_tricks_available"] == 1
    assert telemetry["candidate_tricks"][0]["trick_id"] == "trick.test.metric_bound_clipping"
    assert telemetry["high_risk_tricks_suppressed"] == 1
    assert telemetry["suppressed_high_risk_tricks"][0]["trick_id"] == "trick.test.public_lb_probe"
    assert telemetry["tricks_consulted_by_architect"] is False
    assert telemetry["tricks_used_in_plan"] == []


def test_build_trick_telemetry_stays_closed_for_competitive_cases() -> None:
    module = _validation_module()
    telemetry = module.build_trick_telemetry(
        prompt="tabular regression",
        title="Regression Challenge",
        solution_summary="",
        matches=[],
        assessment="competitive",
        base_assessment="competitive",
        evaluation={"technique_coverage": 1.0, "missing_techniques": []},
        base_evaluation=None,
        delta_plan={},
        trick_retriever=SolutionTrickRetriever([]),
    )

    assert telemetry == {
        "novel_cdg_required": False,
        "candidate_tricks_available": 0,
        "high_risk_tricks_suppressed": 0,
        "tricks_consulted_by_architect": False,
        "tricks_used_in_plan": [],
        "candidate_tricks": [],
        "suppressed_high_risk_tricks": [],
    }

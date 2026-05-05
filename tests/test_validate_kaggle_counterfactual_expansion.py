from __future__ import annotations

import importlib.util
from pathlib import Path

from sciona.principal.expansion_assets import clear_local_expansion_asset_caches


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


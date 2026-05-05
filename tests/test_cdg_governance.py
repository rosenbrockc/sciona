from __future__ import annotations

from copy import deepcopy

from sciona.architect.cdg_governance import (
    CDGGovernanceDecision,
    review_new_base_cdg,
)
from sciona.architect.solution_index import SolutionTemplate, SolutionTemplateIndex
from sciona.principal.expansion_assets import clear_local_expansion_asset_caches


def _stage(stage_id: str, name: str, concept_type: str, description: str = "") -> dict:
    return {
        "stage_id": stage_id,
        "name": name,
        "description": description or name,
        "concept_type": concept_type,
        "inputs": [{"name": f"{stage_id}_input"}],
        "outputs": [{"name": f"{stage_id}_output"}],
    }


def _base_ml_cdg() -> dict:
    return {
        "name": "Base ML Model Selection",
        "family": "ml_model_selection",
        "paradigm": "supervised_learning",
        "summary": "Fit and blend tabular models.",
        "inputs": [{"name": "features"}],
        "outputs": [{"name": "predictions"}],
        "stages": [
            _stage("load_features", "Load features", "data_assembly"),
            _stage("model_training", "Model training", "model_training", "fit estimator"),
            _stage("validation_scoring", "Validation scoring", "analysis"),
            _stage("prediction_ensemble", "Prediction ensemble", "analysis", "combine predictions"),
            _stage("submission_formatting", "Submission formatting", "data_assembly"),
        ],
        "edges": [
            {"source": "load_features", "target": "model_training"},
            {"source": "model_training", "target": "validation_scoring"},
            {"source": "validation_scoring", "target": "prediction_ensemble"},
            {"source": "prediction_ensemble", "target": "submission_formatting"},
        ],
    }


def _template(name: str, raw_cdg: dict) -> SolutionTemplate:
    stages = raw_cdg.get("stages", [])
    return SolutionTemplate(
        name=name,
        family=raw_cdg.get("family", ""),
        paradigm=raw_cdg.get("paradigm", ""),
        summary=raw_cdg.get("summary", ""),
        stage_names=[stage.get("stage_id", "") for stage in stages],
        stage_descriptions=" ".join(stage.get("description", "") for stage in stages),
        raw_cdg=raw_cdg,
    )


def _index() -> SolutionTemplateIndex:
    return SolutionTemplateIndex([_template("base_ml_model_selection", _base_ml_cdg())])


def test_governance_rejects_exact_duplicate_base_cdg() -> None:
    report = review_new_base_cdg(deepcopy(_base_ml_cdg()), _index())

    assert report.decision == CDGGovernanceDecision.REJECT_DUPLICATE
    assert not report.should_accept_base
    assert report.best_existing_template == "base_ml_model_selection"
    assert report.reviews[0].similarity.structural_similarity >= 0.94


def test_governance_flags_mostly_isomorphic_cdg_with_small_delta() -> None:
    clear_local_expansion_asset_caches()
    candidate = deepcopy(_base_ml_cdg())
    candidate["name"] = "Candidate with CV Stacking"
    candidate["stages"] = [
        *candidate["stages"][:3],
        _stage(
            "kfold_ensemble",
            "K-fold cross validated ensemble",
            "analysis",
            "average out-of-fold estimator predictions",
        ),
        _stage(
            "stacking_meta_learner",
            "Stacking meta learner",
            "analysis",
            "train a meta learner on fold predictions",
        ),
        *candidate["stages"][3:],
    ]

    report = review_new_base_cdg(candidate, _index(), max_delta_operations=2)

    assert report.decision == CDGGovernanceDecision.FLAG_BASE_PLUS_DELTA
    assert not report.should_accept_base
    best = report.reviews[0]
    assert best.base_template == "base_ml_model_selection"
    assert best.operation_rule_names == (
        "apply_kfold_ensemble",
        "apply_stacking_ensemble",
    )
    assert best.missing_terms == (
        "K-fold cross validated ensemble",
        "Stacking meta learner",
    )


def test_governance_accepts_distinct_output_contract_as_true_novel() -> None:
    candidate = {
        "name": "Novel graph ranking",
        "family": "graph_optimization",
        "paradigm": "search",
        "summary": "Construct a graph and return ranked paths.",
        "inputs": [{"name": "graph_edges"}],
        "outputs": [{"name": "ranked_paths"}],
        "stages": [
            _stage("build_graph", "Build graph", "graph_traversal"),
            _stage("score_paths", "Score paths", "graph_optimization"),
            _stage("rank_paths", "Rank paths", "sorting"),
        ],
    }

    report = review_new_base_cdg(candidate, _index())

    assert report.decision == CDGGovernanceDecision.ACCEPT_TRUE_NOVEL
    assert report.should_accept_base


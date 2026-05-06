from __future__ import annotations

import json

from sciona.architect.expansion_gap_mining import (
    load_validation_results,
    mine_expansion_gaps,
)
from sciona.principal.expansion_assets import clear_local_expansion_asset_caches
from sciona.principal.trick_retrieval import SolutionTrick, SolutionTrickRetriever


def _result(
    competition_id: str,
    missing: list[str],
    *,
    assessment: str = "partial",
    family: str = "ml_model_selection",
    paradigm: str = "supervised_learning",
) -> dict:
    return {
        "competition_id": competition_id,
        "title": competition_id.replace("-", " ").title(),
        "assessment": assessment,
        "template_matches": [
            {
                "template": f"{family}_template",
                "family": family,
                "paradigm": paradigm,
            }
        ],
        "evaluation": {
            "coverage_source": "llm_semantic",
            "covered_techniques": ["fit estimator"],
            "missing_techniques": missing,
        },
    }


def test_gap_mining_clusters_reusable_missing_techniques() -> None:
    clear_local_expansion_asset_caches()

    report = mine_expansion_gaps(
        [
            _result("comp-a", ["synthetic vortex branch"]),
            _result("comp-b", ["vortex synthetic branch"]),
            _result("comp-c", ["rare one off calibration"]),
        ],
        min_support=2,
        similarity_threshold=0.34,
    )

    reusable = [
        cluster
        for cluster in report.clusters
        if cluster.recommended_action == "candidate_reusable_operation"
    ]
    one_off = [
        cluster
        for cluster in report.clusters
        if cluster.recommended_action == "defer_one_off"
    ]

    assert report.total_results == 3
    assert report.included_results == 3
    assert report.occurrence_count == 3
    assert reusable
    assert reusable[0].support == 2
    assert set(reusable[0].competitions) == {"comp-a", "comp-b"}
    assert one_off[0].support == 1


def test_gap_mining_marks_existing_expansion_assets() -> None:
    clear_local_expansion_asset_caches()

    report = mine_expansion_gaps(
        [
            _result(
                "comp-a",
                ["k-fold cross validated ensemble", "stacking meta learner"],
            )
        ],
        min_support=2,
    )

    existing = [
        cluster
        for cluster in report.clusters
        if cluster.recommended_action == "covered_by_existing_operation"
    ]

    assert existing
    assert existing[0].existing_asset_family == "ml_model_selection"
    existing_rules = {
        rule_name
        for cluster in existing
        for rule_name in cluster.existing_operation_rule_names
    }
    assert existing_rules >= {"apply_kfold_ensemble", "apply_stacking_ensemble"}


def test_gap_mining_uses_base_evaluation_when_adapted_results_exist() -> None:
    result = _result("comp-a", ["base missing"], assessment="partial")
    result["base_evaluation"] = {
        "coverage_source": "keyword_heuristic",
        "missing_techniques": ["base missing"],
    }
    result["evaluation"] = {
        "coverage_source": "keyword_heuristic_plus_delta_counterfactual",
        "missing_techniques": [],
    }

    report = mine_expansion_gaps([result], min_support=2)

    assert report.occurrence_count == 1
    assert report.clusters[0].representative_terms == ("base missing",)


def test_gap_mining_reports_matching_solution_tricks() -> None:
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
                tags=("metric", "clipping"),
            ),
            SolutionTrick(
                trick_id="trick.test.public_lb_probe",
                name="Public leaderboard probing",
                kind="public_lb_overfit_risk",
                status="cataloged",
                risk_level="high",
                generalization_level="competition_specific",
                summary="Probe public leaderboard thresholds.",
                tags=("leaderboard", "thresholding"),
            ),
        ]
    )

    report = mine_expansion_gaps(
        [_result("comp-a", ["metric-bound clipping and leaderboard thresholding"])],
        min_support=2,
        trick_retriever=retriever,
    )

    cluster = report.clusters[0]
    assert cluster.candidate_trick_ids == ("trick.test.metric_bound_clipping",)
    assert cluster.high_risk_trick_ids == ("trick.test.public_lb_probe",)
    assert report.trick_candidate_cluster_count == 1
    assert report.high_risk_trick_cluster_count == 1
    as_dict = report.to_dict()
    assert as_dict["trick_candidate_cluster_count"] == 1
    assert as_dict["clusters"][0]["candidate_trick_ids"] == [
        "trick.test.metric_bound_clipping"
    ]


def test_load_validation_results_accepts_lists_and_wrapped_results(tmp_path) -> None:
    list_path = tmp_path / "list.json"
    wrapped_path = tmp_path / "wrapped.json"
    list_path.write_text(json.dumps([_result("comp-a", ["missing a"])]))
    wrapped_path.write_text(json.dumps({"results": [_result("comp-b", ["missing b"])]}))

    rows = load_validation_results([list_path, wrapped_path])

    assert [row["competition_id"] for row in rows] == ["comp-a", "comp-b"]

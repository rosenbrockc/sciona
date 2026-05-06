from __future__ import annotations

import json

from sciona.architect.validation_followup import (
    build_validation_followup_report,
    build_validation_followup_report_from_paths,
)


def _result(
    competition_id: str,
    *,
    assessment: str = "divergent",
    base_assessment: str = "divergent",
    family: str = "tabular_classification",
    missing: list[str] | None = None,
    candidate_tricks: list[str] | None = None,
    high_risk_tricks: list[str] | None = None,
) -> dict:
    candidate_tricks = candidate_tricks or []
    high_risk_tricks = high_risk_tricks or []
    return {
        "competition_id": competition_id,
        "title": competition_id.replace("-", " ").title(),
        "assessment": assessment,
        "base_assessment": base_assessment,
        "assessment_without_tricks": assessment,
        "assessment_with_trick_availability": (
            f"{assessment}+trick_available"
            if candidate_tricks
            else f"{assessment}+high_risk_trick_suppressed"
            if high_risk_tricks
            else assessment
        ),
        "rescued_by_expansion": assessment != base_assessment,
        "template_matches": [
            {
                "template": f"{family}_template",
                "family": family,
                "paradigm": "supervised_learning",
            }
        ],
        "evaluation": {
            "coverage_source": "keyword_heuristic",
            "missing_techniques": missing or [],
        },
        "trick_telemetry": {
            "candidate_tricks": [{"trick_id": trick_id} for trick_id in candidate_tricks],
            "suppressed_high_risk_tricks": [
                {"trick_id": trick_id} for trick_id in high_risk_tricks
            ],
        },
    }


def test_validation_followup_builds_trick_tickets_and_divergent_buckets() -> None:
    report = build_validation_followup_report(
        [
            _result(
                "comp-a",
                missing=["metric-bound clipping"],
                candidate_tricks=["trick.test.metric_bound_clipping"],
            ),
            _result(
                "comp-b",
                missing=["metric bound clipping"],
                candidate_tricks=["trick.test.metric_bound_clipping"],
            ),
            _result(
                "comp-c",
                assessment="partial",
                base_assessment="divergent",
                family="semantic_segmentation",
                missing=["test-time augmentation"],
                high_risk_tricks=["trick.test.public_lb_probe"],
            ),
        ],
        similarity_threshold=0.34,
    )

    assert report.total_results == 3
    assert report.assessment_counts == {"divergent": 2, "partial": 1}
    assert report.base_assessment_counts == {"divergent": 3}
    assert report.rescued_by_expansion_count == 1
    assert report.trick_review_ticket_count == 3
    assert report.trick_review_tickets[0].competition_id == "comp-a"
    assert report.trick_review_tickets[0].recommended_action == (
        "review_for_expansion_refinement_or_metadata"
    )
    assert report.trick_review_tickets[-1].recommended_action == (
        "keep_suppressed_and_review_for_detection_only"
    )
    assert report.divergent_family_counts == {"tabular_classification": 2}
    assert report.divergent_gap_report.occurrence_count == 2


def test_validation_followup_loads_paths(tmp_path) -> None:
    path = tmp_path / "results.json"
    path.write_text(
        json.dumps(
            [
                _result(
                    "comp-a",
                    missing=["rare missing"],
                    candidate_tricks=["trick.test.metric_bound_clipping"],
                )
            ]
        )
    )

    report = build_validation_followup_report_from_paths([str(path)])

    assert report.total_results == 1
    assert report.trick_review_ticket_count == 1
    assert report.to_dict()["trick_review_tickets"][0]["competition_id"] == "comp-a"

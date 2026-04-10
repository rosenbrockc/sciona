from __future__ import annotations

from sciona.principal.runtime_usability import build_runtime_usability_assessment
from sciona.usability import UsabilityScope


def test_build_runtime_usability_assessment_emits_first_class_artifact() -> None:
    assessment = build_runtime_usability_assessment(
        {
            "runtime_context": {"primary_stream_id": "generic"},
            "telemetry_summary": {
                "signal": {"count": 10.0, "std": 1.0, "mean": 2.0},
                "events": {"count": 4.0, "outlier_fraction": 0.25},
            },
            "heuristics": [
                {
                    "heuristic": {"heuristic_id": "quality_instability"},
                    "confidence": 0.8,
                    "source_section": "signal",
                }
            ],
            "heuristic_summary": {
                "heuristic_count": 1,
                "heuristic_ids": ["quality_instability"],
                "max_confidence": 0.8,
            },
        }
    )

    assert assessment.assessment_id == "runtime_usability_assessment"
    assert assessment.heuristic_signature == ["quality_instability"]
    assert assessment.guidance.scope == UsabilityScope.GUIDANCE
    assert assessment.scoring.scope == UsabilityScope.SCORING
    assert assessment.final_benchmark.scope == UsabilityScope.FINAL_BENCHMARK
    assert assessment.usable_for_guidance is True
    assert assessment.usable_for_scoring is True
    assert assessment.usable_for_final_benchmark is True
    assert assessment.final_benchmark.warning_reasons


def test_build_runtime_usability_assessment_reports_missing_runtime_context() -> None:
    assessment = build_runtime_usability_assessment(
        {
            "telemetry_summary": {},
            "heuristics": [],
            "heuristic_summary": {"heuristic_count": 0, "heuristic_ids": [], "max_confidence": 0.0},
        }
    )

    assert assessment.usable_for_guidance is False
    assert assessment.usable_for_scoring is False
    assert assessment.usable_for_final_benchmark is False
    assert assessment.guidance.blocking_reasons[0].code == "required_input_missing"
    assert assessment.scoring.blocking_reasons[0].code == "coverage_insufficient"
    assert assessment.final_benchmark.warning_reasons[0].code in {
        "review_recommended",
        "narrow_support",
    }

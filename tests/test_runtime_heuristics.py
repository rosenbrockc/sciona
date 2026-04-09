from __future__ import annotations

from sciona.heuristics import (
    HeuristicApplicabilityScope,
    HeuristicProducerKind,
)
from sciona.principal.runtime_heuristics import (
    RuntimeHeuristicEvidence,
    derive_runtime_heuristics,
)


def test_derive_runtime_heuristics_from_canonical_telemetry_summary() -> None:
    evidence = {
        "telemetry_summary": {
            "signal": {
                "count": 2000.0,
                "mean": 2.5,
                "std": 3.5,
                "p50": 1.0,
                "discontinuity_count": 3.0,
                "source_key": "signal",
                "stream_id": "ecg",
            },
            "events": {
                "count": 4.0,
                "duration_seconds": 300.0,
                "density_per_minute": 0.8,
                "interval_median_samples": 250.0,
                "interval_mad_samples": 50.0,
                "outlier_fraction": 0.5,
                "source_key": "events",
            },
            "rate": {
                "count": 3.0,
                "mean": 70.0,
                "std": 80.0,
                "p50": 68.0,
                "source_key": "rate",
            },
            "intermediates": {},
            "outputs": {},
        }
    }

    runtime_heuristics = derive_runtime_heuristics(evidence)
    heuristic_ids = {
        observation.heuristic.heuristic_id
        for observation in runtime_heuristics.observations
    }

    assert heuristic_ids == {
        "boundary_discontinuity",
        "interval_instability",
        "quality_instability",
        "density_collapse",
    }
    assert runtime_heuristics.heuristic_summary["heuristic_count"] == 5
    assert runtime_heuristics.heuristic_summary["max_confidence"] > 0.5
    for observation in runtime_heuristics.observations:
        assert observation.heuristic.producer_kind == HeuristicProducerKind.RUNTIME_TRANSFORM
        assert (
            observation.heuristic.applicability_scope
            == HeuristicApplicabilityScope.CROSS_FAMILY
        )
        assert observation.heuristic.supported_action_classes
        assert observation.provenance == "canonical_telemetry_summary"


def test_runtime_heuristic_evidence_round_trips_through_json_contract() -> None:
    evidence = RuntimeHeuristicEvidence.model_validate(
        derive_runtime_heuristics(
            {
                "telemetry_summary": {
                    "signal": {
                        "count": 128.0,
                        "std": 4.0,
                        "p50": 1.0,
                        "discontinuity_count": 1.0,
                    }
                }
            }
        ).model_dump(mode="json")
    )

    assert evidence.observations
    assert evidence.observations[0].heuristic.heuristic_id == "boundary_discontinuity"

from __future__ import annotations

from sciona.heuristics import HeuristicActionClass
from sciona.principal.heuristic_outcomes import (
    extract_heuristic_outcomes,
    extract_heuristic_usability_memory,
    heuristic_action_bonus,
    summarize_heuristic_usability_memory,
    summarize_heuristic_outcomes,
    summarize_runtime_heuristic_evidence,
)


def test_extract_and_summarize_heuristic_outcomes() -> None:
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
        }
    ]

    outcomes = extract_heuristic_outcomes(search_trace)
    summary = summarize_heuristic_outcomes(outcomes)

    assert len(outcomes) == 1
    assert summary["outcome_count"] == 1
    assert summary["positive_outcome_count"] == 1


def test_heuristic_action_bonus_requires_repeated_positive_outcomes() -> None:
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

    bonus = heuristic_action_bonus(
        family="signal_event_rate",
        heuristic_ids=["interval_instability"],
        search_trace=search_trace,
    )

    assert bonus[HeuristicActionClass.INSERT_CORRECTION] == 1


def test_summarize_runtime_heuristic_evidence_keeps_only_compact_fields() -> None:
    summary = summarize_runtime_heuristic_evidence(
        {
            "runtime_context": {"primary_stream_id": "generic"},
            "canonical_runtime_context": {"primary_stream_id": "generic"},
            "telemetry_summary": {"signal": {"count": 10.0}},
            "heuristics": [{"heuristic": {"heuristic_id": "interval_instability"}}],
            "heuristic_summary": {"heuristic_count": 1},
            "usability_assessment": {"assessment_id": "runtime_usability_assessment"},
            "runtime_inputs": {"raw": [1, 2, 3]},
        }
    )

    assert "runtime_inputs" not in summary
    assert summary["runtime_context"]["primary_stream_id"] == "generic"
    assert summary["heuristic_summary"]["heuristic_count"] == 1


def test_summarize_runtime_heuristic_evidence_refreshes_stale_usability() -> None:
    summary = summarize_runtime_heuristic_evidence(
        {
            "runtime_context": {"primary_stream_id": "generic"},
            "canonical_runtime_context": {"primary_stream_id": "generic"},
            "telemetry_summary": {
                "signal": {"count": 10.0, "mean": 0.0},
                "events": {"count": 5.0, "interval_median_samples": 1.0},
                "rate": {"count": 5.0, "mean": 80.0},
                "outputs": {},
            },
            "heuristics": [
                {
                    "heuristic": {"heuristic_id": "interval_instability"},
                    "confidence": 0.8,
                    "source_section": "rate",
                }
            ],
            "heuristic_summary": {
                "heuristic_count": 1,
                "heuristic_ids": ["interval_instability"],
                "max_confidence": 0.8,
            },
            "execution_summary": {
                "process_returncode": 0,
                "loss_is_finite": True,
                "trace_support_present": True,
                "output_support_present": True,
                "soft_accepted_nonzero_exit": False,
            },
            "usability_assessment": {
                "assessment_id": "runtime_usability_assessment",
                "usable_for_guidance": True,
                "usable_for_scoring": False,
                "usable_for_final_benchmark": False,
            },
        }
    )

    assert summary["execution_summary"]["output_support_present"] is True
    assert summary["usability_assessment"]["usable_for_scoring"] is True
    assert summary["usability_assessment"]["usable_for_final_benchmark"] is True


def test_summarize_runtime_heuristic_evidence_keeps_compact_cohort_gating_data() -> None:
    summary = summarize_runtime_heuristic_evidence(
        {
            "heuristic_cohort": {
                "cohort_size": 5,
                "evaluated_member_count": 2,
                "attempted_member_count": 5,
                "heuristics": {
                    "interval_instability": {
                        "member_count": 2,
                        "coverage_fraction": 1.0,
                    }
                },
                "gating_heuristics": {
                    "quality_instability": {
                        "member_count": 3,
                        "coverage_fraction": 1.0,
                    }
                },
                "usability": {
                    "proposal_basis": "scoring_usable_members",
                    "unscoreable_member_count": 3,
                },
                "excluded_members": [
                    {
                        "member_label": "night_3",
                        "tracker_value": "heuristic_cohort_5_003",
                        "loss": 1e12,
                        "heuristics": [
                            {
                                "heuristic": {"heuristic_id": "quality_instability"},
                                "confidence": 0.7,
                                "source_section": "signal",
                            }
                        ],
                        "usability": {
                            "usable_for_guidance": True,
                            "usable_for_scoring": False,
                            "usable_for_final_benchmark": False,
                            "scope_exclusions": {"scoring": ["coverage_insufficient"]},
                        },
                    }
                ],
            }
        }
    )

    cohort = summary["heuristic_cohort"]
    assert cohort["usability"]["proposal_basis"] == "scoring_usable_members"
    assert cohort["gating_heuristics"]["quality_instability"]["member_count"] == 3
    assert cohort["excluded_members"][0]["heuristics"][0]["heuristic_id"] == "quality_instability"
    assert cohort["excluded_members"][0]["usability"]["usable_for_scoring"] is False


def test_extract_heuristic_usability_memory_tracks_scopes_and_loss_delta() -> None:
    evidence = {
        "runtime_context": {"primary_stream_id": "generic"},
        "canonical_runtime_context": {"primary_stream_id": "generic"},
        "telemetry_summary": {
            "signal": {"count": 10.0, "mean": 0.5},
            "rate": {"count": 5.0, "mean": 80.0},
            "outputs": {},
        },
        "heuristics": [
            {
                "heuristic": {"heuristic_id": "interval_instability"},
                "confidence": 0.8,
                "source_section": "events",
            }
        ],
        "heuristic_summary": {
            "heuristic_count": 1,
            "heuristic_ids": ["interval_instability"],
            "max_confidence": 0.8,
        },
        "execution_summary": {
            "process_returncode": 0,
            "loss_is_finite": True,
            "trace_support_present": True,
            "output_support_present": True,
            "soft_accepted_nonzero_exit": False,
        },
    }
    runtime_evidence = {
        **evidence,
        "usability_assessment": {
            "assessment_id": "runtime_usability_assessment",
            "family": "generic_records",
            "task_intent": "runtime_artifact_emission",
            "heuristic_signature": ["interval_instability"],
            "required_contracts_checked": [
                "runtime_context",
                "canonical_runtime_context",
                "telemetry_summary",
                "heuristics",
                "heuristic_summary",
            ],
            "usable_for_guidance": True,
            "usable_for_scoring": True,
            "usable_for_final_benchmark": True,
            "confidence": 0.8,
            "uncertainty_notes": [],
            "guidance": {
                "scope": "guidance",
                "usable": True,
                "confidence": 0.8,
                "blocking_reasons": [],
                "warning_reasons": [
                    {
                        "kind": "warning",
                        "code": "review_recommended",
                        "summary": "Heuristic support is present but still benefits from review.",
                        "related_heuristic_ids": ["interval_instability"],
                        "confidence": 0.8,
                        "uncertainty_notes": [],
                        "provenance": [
                            {
                                "kind": "runtime_assessor",
                                "source_id": "heuristic_summary",
                                "note": "Heuristic support is present but still benefits from review.",
                            }
                        ],
                    }
                ],
            },
            "scoring": {
                "scope": "scoring",
                "usable": True,
                "confidence": 0.8,
                "blocking_reasons": [],
                "warning_reasons": [],
            },
            "final_benchmark": {
                "scope": "final_benchmark",
                "usable": True,
                "confidence": 0.8,
                "blocking_reasons": [],
                "warning_reasons": [],
            },
        },
    }
    trial_history = [
        {
            "trial": 1,
            "planning_artifact": {"family_hint": "generic_records"},
            "structure": {"topo_hash": "abc"},
            "runtime_evidence": summarize_runtime_heuristic_evidence(runtime_evidence),
            "proposal_selection": {
                "baseline_loss": 10.0,
                "selected": "expansion",
                "candidates": [
                    {
                        "label": "expansion",
                        "family": "generic_records",
                        "loss": 8.5,
                        "evidence": {
                            "heuristic_ids": ["interval_instability"],
                            "candidate_action_classes": ["insert_correction"],
                        },
                    }
                ],
            },
        }
    ]

    records = extract_heuristic_usability_memory(trial_history)
    summary = summarize_heuristic_usability_memory(records)

    assert len(records) == 1
    record = records[0]
    assert record.selected is True
    assert record.loss_delta == 1.5
    assert record.heuristic_signature == ["interval_instability"]
    assert record.usability_assessment_id == "runtime_usability_assessment"
    assert record.usable_for_final_benchmark is True
    assert record.usability_scopes["guidance"].scope == "guidance"
    assert record.usability_scopes["guidance"].usable is True
    assert record.context["structure"]["topo_hash"] == "abc"
    assert record.context["runtime_evidence"]["heuristic_summary"]["heuristic_count"] == 1
    assert summary["memory_count"] == 1
    assert summary["positive_memory_count"] == 1
    assert summary["selected_memory_count"] == 1
    assert summary["heuristic_signatures"]["interval_instability"]["record_count"] == 1
    assert summary["heuristic_signatures"]["interval_instability"]["max_loss_delta"] == 1.5

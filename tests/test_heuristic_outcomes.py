from __future__ import annotations

from sciona.heuristics import HeuristicActionClass
from sciona.principal.heuristic_outcomes import (
    extract_heuristic_outcomes,
    heuristic_action_bonus,
    summarize_heuristic_outcomes,
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

"""Focused tests for search-discipline and benchmark-policy contracts."""

from __future__ import annotations

from sciona.principal.search_policy import (
    enforce_anti_shortcut_policy,
    evaluate_behavioral_benchmark_policy,
    summarize_proposal_selection,
    summarize_search_discipline,
    validate_required_benchmark_artifacts,
)


def test_validate_required_benchmark_artifacts_reports_missing_keys() -> None:
    report = validate_required_benchmark_artifacts(
        {
            "planning_artifact": {},
            "trial_history": [],
        }
    )

    assert report.passed is False
    assert "missing_artifact:skeleton_asset" in report.violations
    assert "missing_artifact:final_candidate" in report.violations


def test_anti_shortcut_policy_rejects_undeclared_shortcuts() -> None:
    report = enforce_anti_shortcut_policy(
        {
            "shortcut_flags": {
                "curated_signal_event_rate_shortcut": True,
                "direct_baseline_only": False,
            },
            "declared_shortcuts": [],
        }
    )

    assert report.passed is False
    assert report.violations == ("undeclared_shortcut:curated_signal_event_rate_shortcut",)


def test_anti_shortcut_policy_warns_for_declared_shortcuts() -> None:
    report = enforce_anti_shortcut_policy(
        {
            "shortcut_flags": {"declared_operational_fast_path": True},
            "declared_shortcuts": ["declared_operational_fast_path"],
        }
    )

    assert report.passed is True
    assert report.warnings == ("declared_shortcut:declared_operational_fast_path",)


def test_behavioral_policy_accepts_semantically_valid_alternative() -> None:
    report = evaluate_behavioral_benchmark_policy(
        {
            "family": "signal_event_rate",
            "ground_truth_coverage": 1.0,
            "used_real_assets": True,
            "executable": True,
            "matched_primitives": [
                "ageoa.biosppy.ecg.bandpass_filter",
                "ageoa.biosppy.ecg.r_peak_detection",
                "ageoa.biosppy.ecg.heart_rate_computation",
            ],
        },
        allowed_families={"signal_event_rate", "signal_detect_measure"},
    )

    assert report.passed is True


def test_behavioral_policy_rejects_non_executable_shortcut_run() -> None:
    report = evaluate_behavioral_benchmark_policy(
        {
            "family": "signal_event_rate",
            "ground_truth_coverage": 0.67,
            "used_real_assets": False,
            "executable": False,
        },
        allowed_families={"signal_event_rate"},
    )

    assert report.passed is False
    assert "insufficient_ground_truth_coverage" in report.violations
    assert "real_assets_not_exercised" in report.violations
    assert "non_executable_candidate" in report.violations


def test_search_discipline_summary_counts_behavioral_signals() -> None:
    summary = summarize_search_discipline(
        [
            {
                "expansion": {"diagnostic_count": 2, "applied": False},
                "admissibility": {"decision_count": 1},
                "reused_cached_evaluation": False,
                "error": "",
            },
            {
                "expansion": {"diagnostic_count": 1, "applied": True},
                "admissibility": {"decision_count": 2},
                "reused_cached_evaluation": True,
                "error": "Trial pruned early",
            },
        ]
    )

    assert summary.trial_count == 2
    assert summary.expansion_attempts == 2
    assert summary.admissibility_decisions == 3
    assert summary.pruned_trials == 1
    assert summary.reused_cached_evaluations == 1


def test_proposal_selection_summary_handles_typed_nested_records() -> None:
    class _Candidate:
        def __init__(self, label: str, loss: float) -> None:
            self.label = label
            self.loss = loss

    class _ProposalSelection:
        def __init__(self) -> None:
            self.selected_proposal = "expansion"
            self.proposal_candidates = [
                _Candidate("expansion", 8.5),
                _Candidate("local_mutation", 9.0),
            ]
            self.proposal_baseline_loss = 10.0
            self.proposal_improvement = 1.5

        def model_dump(self, mode: str = "json") -> dict[str, object]:
            return {
                "selected_proposal": self.selected_proposal,
                "proposal_candidates": [
                    {"label": candidate.label, "loss": candidate.loss}
                    for candidate in self.proposal_candidates
                ],
                "proposal_baseline_loss": self.proposal_baseline_loss,
                "proposal_improvement": self.proposal_improvement,
            }

    summary = summarize_proposal_selection(
        [
            {
                "proposal_selection": _ProposalSelection(),
                "expansion": {"applied": True},
            },
            {
                "proposal_selection": {
                    "selected": "",
                    "candidates": [],
                    "skipped_due_to_admissibility": True,
                }
            },
        ]
    )

    assert summary.trial_count == 2
    assert summary.proposal_selection_trials == 2
    assert summary.selected_trials == 1
    assert summary.rejected_trials == 1
    assert summary.skipped_due_to_admissibility_trials == 1
    assert summary.selected_proposal_counts == {"expansion": 1}
    assert summary.proposal_selection_labels == ("expansion", "local_mutation")
    assert summary.mean_selected_proposal_improvement == 1.5
    assert summary.best_selected_proposal_improvement == 1.5

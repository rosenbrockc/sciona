"""Search-discipline and benchmark-policy contracts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class SearchDisciplineSummary:
    """Compact summary of how a run searched rather than only how it ended."""

    trial_count: int
    expansion_attempts: int
    admissibility_decisions: int
    pruned_trials: int
    reused_cached_evaluations: int


@dataclass(frozen=True)
class ProposalSelectionSummary:
    """Compact summary of proposal-selection behavior in trial history."""

    trial_count: int
    proposal_selection_trials: int
    selected_trials: int
    rejected_trials: int
    skipped_due_to_admissibility_trials: int
    selected_proposal_counts: dict[str, int] = field(default_factory=dict)
    proposal_selection_labels: tuple[str, ...] = ()
    mean_selected_proposal_improvement: float = 0.0
    best_selected_proposal_improvement: float = 0.0


def _coerce_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        if isinstance(dumped, dict):
            return dict(dumped)
    if hasattr(value, "__dict__"):
        return {
            key: val
            for key, val in vars(value).items()
            if not key.startswith("_")
        }
    return {}


def _normalize_proposal_selection(entry: dict[str, Any]) -> dict[str, Any]:
    proposal = _coerce_mapping(entry.get("proposal_selection", {}))
    if not proposal:
        return {}

    selected = proposal.get("selected")
    if selected in (None, ""):
        selected = proposal.get("selected_proposal")
    if selected in (None, ""):
        selected = proposal.get("proposal_selected")
    if selected in (None, ""):
        selected = proposal.get("label")
    decision = proposal.get("decision")
    if selected in (None, "") and isinstance(decision, dict):
        selected = decision.get("selected")
    selected = str(selected or "")

    candidates = proposal.get("candidates", proposal.get("proposal_candidates", []))
    if not isinstance(candidates, list):
        candidates = []

    candidate_labels: list[str] = []
    selected_loss: float | None = None
    for candidate in candidates:
        candidate_map = _coerce_mapping(candidate)
        if not candidate_map:
            continue
        label = str(
            candidate_map.get("label")
            or candidate_map.get("proposal_type")
            or candidate_map.get("name")
            or ""
        ).strip()
        if label:
            candidate_labels.append(label)
        if selected and label == selected:
            try:
                selected_loss = float(candidate_map.get("loss", 0.0) or 0.0)
            except (TypeError, ValueError):
                selected_loss = None

    baseline_loss = proposal.get("baseline_loss", proposal.get("proposal_baseline_loss"))
    try:
        baseline_loss_value = (
            float(baseline_loss) if baseline_loss is not None else None
        )
    except (TypeError, ValueError):
        baseline_loss_value = None

    selected_improvement = proposal.get("proposal_improvement")
    if selected_improvement is None and baseline_loss_value is not None and selected_loss is not None:
        selected_improvement = baseline_loss_value - selected_loss
    try:
        selected_improvement_value = (
            float(selected_improvement) if selected_improvement is not None else None
        )
    except (TypeError, ValueError):
        selected_improvement_value = None

    return {
        "selected": selected,
        "candidate_labels": candidate_labels,
        "candidate_count": len(candidates),
        "selected_improvement": selected_improvement_value,
        "skipped_due_to_admissibility": bool(
            proposal.get("skipped_due_to_admissibility")
        ),
        "skip_reason": str(proposal.get("skip_reason", "") or ""),
        "proposal": proposal,
    }


def summarize_proposal_selection(
    trial_history: list[dict[str, Any]],
) -> ProposalSelectionSummary:
    """Summarize proposal-selection behavior from persisted trial history."""
    proposal_selection_trials = 0
    selected_trials = 0
    rejected_trials = 0
    skipped_due_to_admissibility_trials = 0
    selected_proposal_counts: dict[str, int] = {}
    proposal_selection_labels: list[str] = []
    improvements: list[float] = []

    for entry in trial_history:
        if not isinstance(entry, dict):
            continue
        proposal = _normalize_proposal_selection(entry)
        if not proposal:
            continue

        proposal_selection_trials += 1
        proposal_selection_labels.extend(
            label for label in proposal.get("candidate_labels", []) if label
        )
        selected = str(proposal.get("selected", "") or "")
        if selected:
            selected_trials += 1
            selected_proposal_counts[selected] = (
                selected_proposal_counts.get(selected, 0) + 1
            )
            improvement = proposal.get("selected_improvement")
            if isinstance(improvement, (int, float)):
                improvements.append(float(improvement))
        else:
            rejected_trials += 1
        if proposal.get("skipped_due_to_admissibility"):
            skipped_due_to_admissibility_trials += 1

    return ProposalSelectionSummary(
        trial_count=len(trial_history),
        proposal_selection_trials=proposal_selection_trials,
        selected_trials=selected_trials,
        rejected_trials=rejected_trials,
        skipped_due_to_admissibility_trials=skipped_due_to_admissibility_trials,
        selected_proposal_counts=dict(sorted(selected_proposal_counts.items())),
        proposal_selection_labels=tuple(sorted(set(proposal_selection_labels))),
        mean_selected_proposal_improvement=(
            float(sum(improvements) / len(improvements)) if improvements else 0.0
        ),
        best_selected_proposal_improvement=(
            float(max(improvements)) if improvements else 0.0
        ),
    )


@dataclass(frozen=True)
class BenchmarkPolicyReport:
    """Structured benchmark-policy outcome."""

    passed: bool
    violations: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)


def summarize_search_discipline(trial_history: list[dict[str, Any]]) -> SearchDisciplineSummary:
    """Summarize high-level search behavior from persisted trial history."""
    expansion_attempts = 0
    admissibility_decisions = 0
    pruned_trials = 0
    reused_cached = 0
    for entry in trial_history:
        expansion = entry.get("expansion", {})
        if isinstance(expansion, dict) and (
            expansion.get("diagnostic_count", 0) or expansion.get("applied")
        ):
            expansion_attempts += 1
        admissibility = entry.get("admissibility", {})
        if isinstance(admissibility, dict):
            admissibility_decisions += int(admissibility.get("decision_count", 0) or 0)
        if "pruned" in str(entry.get("error", "")).lower():
            pruned_trials += 1
        if bool(entry.get("reused_cached_evaluation")):
            reused_cached += 1
    return SearchDisciplineSummary(
        trial_count=len(trial_history),
        expansion_attempts=expansion_attempts,
        admissibility_decisions=admissibility_decisions,
        pruned_trials=pruned_trials,
        reused_cached_evaluations=reused_cached,
    )


def validate_required_benchmark_artifacts(
    artifacts: dict[str, Any],
    *,
    required_keys: tuple[str, ...] = (
        "planning_artifact",
        "skeleton_asset",
        "trial_history",
        "final_candidate",
        "runtime_context",
    ),
) -> BenchmarkPolicyReport:
    """Require the minimum artifact set needed to judge search behavior."""
    missing = [key for key in required_keys if key not in artifacts]
    return BenchmarkPolicyReport(
        passed=not missing,
        violations=tuple(f"missing_artifact:{key}" for key in missing),
        details={"required_keys": list(required_keys)},
    )


def enforce_anti_shortcut_policy(
    benchmark_metadata: dict[str, Any],
) -> BenchmarkPolicyReport:
    """Reject undeclared shortcut paths for benchmark/e2e runs."""
    shortcut_flags = benchmark_metadata.get("shortcut_flags", {})
    declared_shortcuts = benchmark_metadata.get("declared_shortcuts", [])
    if not isinstance(shortcut_flags, dict):
        shortcut_flags = {}
    if not isinstance(declared_shortcuts, list):
        declared_shortcuts = []

    violations: list[str] = []
    warnings: list[str] = []
    for key, enabled in sorted(shortcut_flags.items()):
        if not enabled:
            continue
        if key in declared_shortcuts:
            warnings.append(f"declared_shortcut:{key}")
        else:
            violations.append(f"undeclared_shortcut:{key}")
    return BenchmarkPolicyReport(
        passed=not violations,
        violations=tuple(violations),
        warnings=tuple(warnings),
        details={"shortcut_flags": dict(shortcut_flags)},
    )


def evaluate_behavioral_benchmark_policy(
    result: dict[str, Any],
    *,
    allowed_families: set[str],
    min_ground_truth_coverage: float = 1.0,
    require_real_assets: bool = True,
) -> BenchmarkPolicyReport:
    """Prefer semantic and behavioral checks over exact scaffold matching."""
    violations: list[str] = []
    details: dict[str, Any] = {}

    family = str(result.get("family", "") or "")
    coverage = float(result.get("ground_truth_coverage", 0.0) or 0.0)
    used_real_assets = bool(result.get("used_real_assets"))
    executable = bool(result.get("executable"))

    details.update(
        {
            "family": family,
            "ground_truth_coverage": coverage,
            "used_real_assets": used_real_assets,
            "executable": executable,
        }
    )

    if family not in allowed_families:
        violations.append(f"disallowed_family:{family}")
    if coverage < min_ground_truth_coverage:
        violations.append("insufficient_ground_truth_coverage")
    if require_real_assets and not used_real_assets:
        violations.append("real_assets_not_exercised")
    if not executable:
        violations.append("non_executable_candidate")

    return BenchmarkPolicyReport(
        passed=not violations,
        violations=tuple(violations),
        details=details,
    )

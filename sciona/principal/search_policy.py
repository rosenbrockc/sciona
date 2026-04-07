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

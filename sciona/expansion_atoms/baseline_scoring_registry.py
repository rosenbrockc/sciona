"""Registry for baseline-path scoring primitives."""

from __future__ import annotations


BASELINE_SCORING_DECLARATIONS = {
    "accumulate_analyzed_time": (
        "sciona.expansion_atoms.runtime_baseline_scoring.accumulate_analyzed_time",
        "np.ndarray, np.ndarray -> tuple[float, bool]",
        "Accumulate analyzed time from a sleep mask aligned to an anchor grid.",
    ),
    "accumulate_prediction_window_time": (
        "sciona.expansion_atoms.runtime_baseline_scoring.accumulate_prediction_window_time",
        "np.ndarray, np.ndarray -> tuple[float, bool]",
        "Accumulate padded prediction-window coverage from component probabilities.",
    ),
    "compute_event_rate_per_hour": (
        "sciona.expansion_atoms.runtime_baseline_scoring.compute_event_rate_per_hour",
        "np.ndarray | int, float -> tuple[float, bool]",
        "Compute an hourly event rate from event labels, intervals, or counts.",
    ),
    "apply_bmi_correction": (
        "sciona.expansion_atoms.runtime_baseline_scoring.apply_bmi_correction",
        "float, float | None -> tuple[float, bool]",
        "Apply the mild-branch BMI correction used by baseline-path scoring.",
    ),
    "score_baseline_path": (
        "sciona.expansion_atoms.runtime_baseline_scoring.score_baseline_path",
        "np.ndarray | int, np.ndarray | int, float, float -> tuple[float, bool]",
        "Score the SQI baseline path into an sAHI-style scalar.",
    ),
    "score_bmi_baseline_path": (
        "sciona.expansion_atoms.runtime_baseline_scoring.score_bmi_baseline_path",
        "np.ndarray | int, np.ndarray | int, float, float, float | None -> tuple[float, bool]",
        "Score the BMI-corrected baseline path into a bAHI-style scalar.",
    ),
    "score_pat_baseline_path": (
        "sciona.expansion_atoms.runtime_baseline_scoring.score_pat_baseline_path",
        "np.ndarray | int, float, float -> tuple[float, bool]",
        "Score the PAT baseline branch into a pAHI-style scalar.",
    ),
}


BASELINE_SCORING_ALTERNATIVES: dict[str, tuple[str, ...]] = {}


def next_baseline_scoring_variant(primitive_name: str) -> str | None:
    """Return the next curated variant for a baseline scoring primitive."""
    variants = BASELINE_SCORING_ALTERNATIVES.get(primitive_name)
    if not variants:
        return None
    return variants[0]

"""Registry for baseline analysis step primitives."""

from __future__ import annotations

BASELINE_STEPS_DECLARATIONS = {
    # --- Mask ---
    "baseline_mask": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_mask",
        "ndarray, ndarray, ndarray -> tuple[ndarray, bool]",
        "Apply zeroing/masking to window data.",
    ),
    # --- Resample ---
    "baseline_resample": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_resample",
        "ndarray, ndarray, ndarray -> tuple[ndarray, bool]",
        "Resample signal to anchor sample rate.",
    ),
    # --- Scale ---
    "baseline_scale_constant": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_scale_constant",
        "ndarray -> tuple[ndarray, bool]",
        "Normalize signal magnitude by constant ratio.",
    ),
    "baseline_scale_wavelet": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_scale_wavelet",
        "ndarray -> tuple[ndarray, bool]",
        "Normalize signal magnitude via wavelet prominence.",
    ),
    # --- Per-Window Fit ---
    "baseline_fit_exp_rise": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_fit_exp_rise",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Fit exponential rise segments.",
    ),
    "baseline_fit_exp_fall": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_fit_exp_fall",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Fit exponential fall segments.",
    ),
    "baseline_fit_sinh_rise": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_fit_sinh_rise",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Fit hyperbolic-sine rise segments.",
    ),
    "baseline_fit_sinh_fall": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_fit_sinh_fall",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Fit hyperbolic-sine fall segments.",
    ),
    # --- Output Transform ---
    "baseline_output_nonzero": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_output_nonzero",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Extract non-zero onset values from fit results.",
    ),
    "baseline_output_clipshift": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_output_clipshift",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Threshold-shift and clip onset values.",
    ),
    "baseline_output_copy": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_output_copy",
        "ndarray, ndarray -> tuple[ndarray, bool]",
        "Pass through onset values unchanged.",
    ),
    # --- Pad ---
    "baseline_pad_constant": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_pad_constant",
        "ndarray, ndarray, ndarray -> tuple[ndarray, bool]",
        "Rectangular padding around onsets.",
    ),
    "baseline_pad_linear": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_pad_linear",
        "ndarray, ndarray, ndarray -> tuple[ndarray, bool]",
        "Linearly decaying padding around onsets.",
    ),
    "baseline_pad_exponential": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_pad_exponential",
        "ndarray, ndarray, ndarray -> tuple[ndarray, bool]",
        "Exponentially decaying padding around onsets.",
    ),
    "baseline_pad_gaussian": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_pad_gaussian",
        "ndarray, ndarray, ndarray -> tuple[ndarray, bool]",
        "Gaussian-shaped padding around onsets.",
    ),
    # --- Normalize ---
    "baseline_normalize_max": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_normalize_max",
        "ndarray -> tuple[ndarray, bool]",
        "Normalize signal by maximum value.",
    ),
    "baseline_normalize_constant": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_normalize_constant",
        "ndarray -> tuple[ndarray, bool]",
        "Normalize signal by fixed constant.",
    ),
    "baseline_normalize_quantile": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_normalize_quantile",
        "ndarray -> tuple[ndarray, bool]",
        "Normalize signal by quantile value.",
    ),
    # --- Regionize ---
    "baseline_regionize": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_regionize",
        "ndarray -> tuple[ndarray, bool]",
        "Threshold signal into discrete event regions.",
    ),
    # --- Combine ---
    "baseline_combine_product": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_combine_product",
        "list[ndarray] -> tuple[ndarray, bool]",
        "Element-wise product of component vectors.",
    ),
    "baseline_combine_convolve": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_combine_convolve",
        "list[ndarray] -> tuple[ndarray, bool]",
        "Sequential convolution of component vectors.",
    ),
    "baseline_combine_weighted": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_combine_weighted",
        "list[ndarray] -> tuple[ndarray, bool]",
        "Weighted element-wise product of component vectors.",
    ),
    "baseline_combine_coherence": (
        "sciona.expansion_atoms.runtime_baseline_steps.baseline_combine_coherence",
        "list[ndarray] -> tuple[ndarray, bool]",
        "Morphological-coherence weighted combination of component vectors.",
    ),
}


BASELINE_ANALYSIS_ALTERNATIVES = {
    "baseline_scale_constant": ("baseline_scale_wavelet",),
    "baseline_scale_wavelet": ("baseline_scale_constant",),
    "baseline_fit_exp_rise": (
        "baseline_fit_exp_fall",
        "baseline_fit_sinh_rise",
        "baseline_fit_sinh_fall",
    ),
    "baseline_fit_exp_fall": (
        "baseline_fit_exp_rise",
        "baseline_fit_sinh_rise",
        "baseline_fit_sinh_fall",
    ),
    "baseline_fit_sinh_rise": (
        "baseline_fit_exp_rise",
        "baseline_fit_exp_fall",
        "baseline_fit_sinh_fall",
    ),
    "baseline_fit_sinh_fall": (
        "baseline_fit_exp_rise",
        "baseline_fit_exp_fall",
        "baseline_fit_sinh_rise",
    ),
    "baseline_output_nonzero": ("baseline_output_clipshift", "baseline_output_copy"),
    "baseline_output_clipshift": ("baseline_output_nonzero", "baseline_output_copy"),
    "baseline_output_copy": ("baseline_output_nonzero", "baseline_output_clipshift"),
    "baseline_pad_constant": (
        "baseline_pad_exponential",
        "baseline_pad_linear",
        "baseline_pad_gaussian",
    ),
    "baseline_pad_exponential": (
        "baseline_pad_constant",
        "baseline_pad_linear",
        "baseline_pad_gaussian",
    ),
    "baseline_pad_linear": (
        "baseline_pad_constant",
        "baseline_pad_exponential",
        "baseline_pad_gaussian",
    ),
    "baseline_pad_gaussian": (
        "baseline_pad_constant",
        "baseline_pad_exponential",
        "baseline_pad_linear",
    ),
    "baseline_normalize_max": (
        "baseline_normalize_constant",
        "baseline_normalize_quantile",
    ),
    "baseline_normalize_constant": (
        "baseline_normalize_max",
        "baseline_normalize_quantile",
    ),
    "baseline_normalize_quantile": (
        "baseline_normalize_max",
        "baseline_normalize_constant",
    ),
    "baseline_combine_product": (
        "baseline_combine_convolve",
        "baseline_combine_weighted",
        "baseline_combine_coherence",
    ),
    "baseline_combine_convolve": (
        "baseline_combine_product",
        "baseline_combine_weighted",
        "baseline_combine_coherence",
    ),
    "baseline_combine_weighted": (
        "baseline_combine_product",
        "baseline_combine_convolve",
        "baseline_combine_coherence",
    ),
    "baseline_combine_coherence": (
        "baseline_combine_product",
        "baseline_combine_convolve",
        "baseline_combine_weighted",
    ),
}


def next_baseline_analysis_variant(primitive_name: str) -> str | None:
    """Return the next curated variant for a baseline analysis primitive."""
    variants = BASELINE_ANALYSIS_ALTERNATIVES.get(primitive_name)
    if not variants:
        return None
    return variants[0]

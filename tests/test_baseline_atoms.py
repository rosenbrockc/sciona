"""Tests for baseline-analysis step atoms and registry declarations."""

from __future__ import annotations

import importlib

import numpy as np
import pytest

from sciona.expansion_atoms.baseline_steps_registry import (
    BASELINE_ANALYSIS_ALTERNATIVES,
    BASELINE_STEPS_DECLARATIONS,
    next_baseline_analysis_variant,
)
from sciona.expansion_atoms.runtime_baseline_steps import (
    baseline_combine_coherence,
    baseline_combine_convolve,
    baseline_combine_product,
    baseline_combine_weighted,
    baseline_fit_exp_fall,
    baseline_fit_exp_rise,
    baseline_fit_sinh_fall,
    baseline_fit_sinh_rise,
    baseline_mask,
    baseline_normalize_constant,
    baseline_normalize_max,
    baseline_normalize_quantile,
    baseline_pad_exponential,
    baseline_pad_gaussian,
    baseline_output_clipshift,
    baseline_output_copy,
    baseline_output_nonzero,
    baseline_pad_constant,
    baseline_pad_linear,
    baseline_regionize,
    baseline_resample,
    baseline_scale_constant,
    baseline_scale_wavelet,
)


class TestBaselineMask:
    def test_zero_mode_zeros_masked_regions(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        result, ok = baseline_mask(signal, np.arange(signal.size), mask)
        assert ok is True
        assert np.array_equal(result, np.array([1.0, 0.0, 3.0, 0.0]))

    def test_slice_mode_returns_compacted_signal(self):
        signal = np.array([1.0, 2.0, 3.0, 4.0])
        mask = np.array([True, False, True, False])
        result, ok = baseline_mask(signal, np.arange(signal.size), mask, zero=False)
        assert ok is True
        assert np.array_equal(result, np.array([1.0, 3.0]))

    def test_all_zero_output_reports_false(self):
        signal = np.zeros(4)
        mask = np.array([True, True, False, False])
        result, ok = baseline_mask(signal, np.arange(signal.size), mask)
        assert result.shape == signal.shape
        assert ok is False


class TestBaselineResample:
    def test_linear_upsample(self):
        signal = np.array([0.0, 10.0, 20.0])
        t = np.array([0.0, 1.0, 2.0])
        anchor = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
        result, ok = baseline_resample(signal, t, anchor)
        assert ok is True
        assert np.allclose(result, np.array([0.0, 5.0, 10.0, 15.0, 20.0]))

    def test_linear_downsample(self):
        signal = np.array([0.0, 10.0, 20.0])
        t = np.array([0.0, 1.0, 2.0])
        anchor = np.array([0.0, 2.0])
        result, ok = baseline_resample(signal, t, anchor)
        assert ok is True
        assert np.allclose(result, np.array([0.0, 20.0]))

    def test_identity_grid_is_unchanged(self):
        signal = np.array([2.0, 4.0, 8.0])
        t = np.array([0.0, 1.0, 2.0])
        result, ok = baseline_resample(signal, t, t)
        assert ok is True
        assert np.allclose(result, signal)


class TestBaselineScaleConstant:
    def test_scales_by_constant_ratio(self):
        signal = np.array([0.0, 0.5, 1.0])
        result, ok = baseline_scale_constant(signal, floor=0.0, ceil=2.0)
        assert ok is True
        assert np.allclose(result, np.array([0.0, 1.0, 2.0]))

    def test_below_floor_returns_copy(self):
        signal = np.array([0.1, 0.2, 0.3])
        result, ok = baseline_scale_constant(signal, floor=0.5, ceil=1.0)
        assert ok is True
        assert np.allclose(result, signal)


class TestBaselineOutputNonzero:
    def test_extracts_nonzero_values(self):
        values = np.array([0.0, 2.0, 0.0, 4.0])
        result, ok = baseline_output_nonzero(values, np.arange(values.size))
        assert ok is True
        assert np.array_equal(result, np.array([2.0, 4.0]))

    def test_discretize_returns_ones(self):
        values = np.array([0.0, 2.0, 0.0, 4.0])
        result, ok = baseline_output_nonzero(
            values,
            np.arange(values.size),
            discretize=True,
        )
        assert ok is True
        assert np.array_equal(result, np.ones(2))

    def test_all_zero_values_return_empty(self):
        values = np.zeros(4)
        result, ok = baseline_output_nonzero(values, np.arange(values.size))
        assert result.size == 0
        assert ok is False


class TestBaselineOutputClipshift:
    def test_threshold_removes_small_values(self):
        values = np.array([0.1, 0.5, 1.5])
        result, ok = baseline_output_clipshift(
            values,
            np.arange(values.size),
            threshold=0.5,
            ceil=1.0,
        )
        assert ok is True
        assert np.allclose(result, np.array([1.0]))

    def test_qscale_normalizes_before_thresholding(self):
        values = np.array([1.0, 2.0, 4.0])
        result, ok = baseline_output_clipshift(
            values,
            np.arange(values.size),
            threshold=0.2,
            ceil=1.0,
            qscale=0.5,
        )
        assert ok is True
        assert np.all(result <= 1.0)
        assert result.size > 0


class TestBaselineOutputCopy:
    def test_returns_copy_and_valid_flag(self):
        values = np.array([0.0, 2.0, 4.0])
        result, ok = baseline_output_copy(values, np.arange(values.size))
        assert ok is True
        assert np.array_equal(result, values)
        assert result is not values


class TestBaselinePadConstant:
    def test_positive_width_accumulates_rectangular_padding(self):
        onsets = np.array([1.0])
        t = np.array([1.0])
        anchor = np.arange(0.0, 5.0)
        result, ok = baseline_pad_constant(onsets, t, anchor, width=2.0)
        assert ok is True
        assert np.array_equal(result, np.array([0.0, 1.0, 1.0, 1.0, 0.0]))

    def test_negative_width_pads_before_onset(self):
        onsets = np.array([2.0])
        t = np.array([3.0])
        anchor = np.arange(0.0, 6.0)
        result, ok = baseline_pad_constant(onsets, t, anchor, width=-2.0)
        assert ok is True
        assert np.array_equal(result, np.array([0.0, 2.0, 2.0, 2.0, 0.0, 0.0]))

    def test_empty_input_is_noop(self):
        result, ok = baseline_pad_constant(
            np.array([]),
            np.array([]),
            np.arange(4.0),
        )
        assert result.shape == (4,)
        assert ok is False


class TestBaselinePadLinear:
    def test_linear_decay_tapers_to_zero(self):
        onsets = np.array([2.0])
        t = np.array([1.0])
        anchor = np.arange(0.0, 5.0)
        result, ok = baseline_pad_linear(onsets, t, anchor, width=2.0)
        assert ok is True
        assert np.allclose(result, np.array([0.0, 2.0, 1.0, 0.0, 0.0]))


class TestBaselineNormalizeMax:
    def test_max_normalization_reaches_one(self):
        signal = np.array([1.0, 2.0, 4.0])
        result, ok = baseline_normalize_max(signal)
        assert ok is True
        assert np.isclose(result.max(), 1.0)
        assert np.allclose(result, np.array([0.25, 0.5, 1.0]))

    def test_non_positive_signal_is_rejected(self):
        signal = np.array([0.0, -1.0, -2.0])
        result, ok = baseline_normalize_max(signal)
        assert ok is False
        assert np.array_equal(result, np.zeros_like(signal))


class TestBaselineNormalizeConstant:
    def test_constant_normalization_clips_to_unit_interval(self):
        signal = np.array([0.0, 1.0, 3.0])
        result, ok = baseline_normalize_constant(signal, value=2.0)
        assert ok is True
        assert np.allclose(result, np.array([0.0, 0.5, 1.0]))


class TestBaselineNormalizeQuantile:
    def test_quantile_normalization_uses_requested_divisor(self):
        signal = np.array([0.0, 1.0, 2.0, 10.0])
        result, ok = baseline_normalize_quantile(signal, q=0.5)
        assert ok is True
        assert np.all(result <= 1.0)
        assert np.isclose(result[2], 1.0)


class TestBaselineRegionize:
    def test_regions_are_labeled_in_order(self):
        signal = np.array([0.0, 0.7, 0.8, 0.1, 0.9, 1.0])
        result, ok = baseline_regionize(signal, threshold=0.5, min_length=1)
        assert ok is True
        assert np.array_equal(result, np.array([0, 1, 1, 0, 2, 2]))

    def test_min_length_filters_short_regions(self):
        signal = np.array([0.0, 0.7, 0.1, 0.8, 0.9])
        result, ok = baseline_regionize(signal, threshold=0.5, min_length=2)
        assert ok is True
        assert np.array_equal(result, np.array([0, 0, 0, 1, 1]))

    def test_empty_signal_is_safe(self):
        result, ok = baseline_regionize(np.array([]))
        assert result.size == 0
        assert ok is False


class TestBaselineRegistry:
    def test_expected_declarations_are_present(self):
        assert set(BASELINE_STEPS_DECLARATIONS) == {
            "baseline_mask",
            "baseline_resample",
            "baseline_scale_constant",
            "baseline_scale_wavelet",
            "baseline_fit_exp_rise",
            "baseline_fit_exp_fall",
            "baseline_fit_sinh_rise",
            "baseline_fit_sinh_fall",
            "baseline_output_nonzero",
            "baseline_output_clipshift",
            "baseline_output_copy",
            "baseline_pad_constant",
            "baseline_pad_linear",
            "baseline_pad_exponential",
            "baseline_pad_gaussian",
            "baseline_normalize_max",
            "baseline_normalize_constant",
            "baseline_normalize_quantile",
            "baseline_regionize",
            "baseline_combine_product",
            "baseline_combine_convolve",
            "baseline_combine_weighted",
            "baseline_combine_coherence",
        }

    @pytest.mark.parametrize("name", sorted(BASELINE_STEPS_DECLARATIONS))
    def test_declaration_fqdns_are_importable(self, name: str):
        fqdn, _sig, _desc = BASELINE_STEPS_DECLARATIONS[name]
        module_name, attr_name = fqdn.rsplit(".", 1)
        module = importlib.import_module(module_name)
        assert hasattr(module, attr_name)

    def test_forward_variant_map_includes_current_alternatives(self):
        assert BASELINE_ANALYSIS_ALTERNATIVES["baseline_scale_constant"] == (
            "baseline_scale_wavelet",
        )
        assert BASELINE_ANALYSIS_ALTERNATIVES["baseline_fit_exp_rise"] == (
            "baseline_fit_exp_fall",
            "baseline_fit_sinh_rise",
            "baseline_fit_sinh_fall",
        )
        assert BASELINE_ANALYSIS_ALTERNATIVES["baseline_pad_constant"] == (
            "baseline_pad_exponential",
            "baseline_pad_linear",
            "baseline_pad_gaussian",
        )
        assert BASELINE_ANALYSIS_ALTERNATIVES["baseline_combine_product"] == (
            "baseline_combine_convolve",
            "baseline_combine_weighted",
            "baseline_combine_coherence",
        )
        assert BASELINE_ANALYSIS_ALTERNATIVES["baseline_output_nonzero"] == (
            "baseline_output_clipshift",
            "baseline_output_copy",
        )
        assert next_baseline_analysis_variant("baseline_output_nonzero") == "baseline_output_clipshift"
        assert next_baseline_analysis_variant("baseline_scale_constant") == "baseline_scale_wavelet"
        assert next_baseline_analysis_variant("baseline_pad_constant") == "baseline_pad_exponential"
        assert next_baseline_analysis_variant("baseline_combine_product") == "baseline_combine_convolve"
        assert next_baseline_analysis_variant("baseline_regionize") is None


class TestBaselineScaleWavelet:
    def test_wavelet_scaling_returns_nontrivial_result(self):
        signal = np.sin(np.linspace(0.0, 6.0 * np.pi, 256)) + 0.25 * np.sin(
            np.linspace(0.0, 24.0 * np.pi, 256)
        )
        result, ok = baseline_scale_wavelet(signal)
        assert ok is True
        assert result.shape == signal.shape
        assert np.any(result > 0.0)

    def test_short_signal_is_rejected(self):
        result, ok = baseline_scale_wavelet(np.array([1.0, 2.0, 3.0]))
        assert ok is False
        assert np.array_equal(result, np.array([1.0, 2.0, 3.0]))


class TestBaselineFitExpRise:
    def test_detects_rising_exponential_segment(self):
        t = np.linspace(0.0, 10.0, 200)
        signal = np.full_like(t, 0.1)
        idx = t >= 4.0
        signal[idx] = -0.8 * np.exp(-0.5 * (t[idx] - 4.0)) + 1.0
        result, ok = baseline_fit_exp_rise(signal, t)
        assert ok is True
        assert np.any(result > 0.0)

    def test_too_short_signal_is_rejected(self):
        t = np.linspace(0.0, 1.0, 3)
        signal = np.array([0.1, 0.2, 0.3])
        result, ok = baseline_fit_exp_rise(signal, t, min_width=4)
        assert ok is False
        assert np.array_equal(result, np.zeros_like(signal))


class TestBaselineFitExpFall:
    def test_detects_falling_exponential_segment(self):
        t = np.linspace(0.0, 10.0, 200)
        signal = np.full_like(t, 1.0)
        idx = t >= 4.0
        signal[idx] = 0.8 * np.exp(-0.5 * (t[idx] - 4.0)) + 0.2
        result, ok = baseline_fit_exp_fall(signal, t)
        assert ok is True
        assert np.any(result > 0.0)

    def test_bad_alignment_is_rejected(self):
        signal = np.array([1.0, 0.9, 0.8])
        t = np.array([0.0, 1.0])
        result, ok = baseline_fit_exp_fall(signal, t)
        assert ok is False
        assert np.array_equal(result, np.zeros_like(signal))


class TestBaselineFitSinhRise:
    def test_detects_rising_sinh_segment(self):
        t = np.linspace(0.0, 6.0, 180)
        signal = np.full_like(t, 0.2)
        idx = t >= 2.5
        signal[idx] = 0.12 * np.sinh(0.8 * (t[idx] - 2.5)) + 0.3
        result, ok = baseline_fit_sinh_rise(signal, t)
        assert ok is True
        assert np.any(result > 0.0)

    def test_too_short_segment_is_rejected(self):
        t = np.linspace(0.0, 1.0, 3)
        signal = np.array([0.1, 0.2, 0.4])
        result, ok = baseline_fit_sinh_rise(signal, t, min_width=4)
        assert ok is False
        assert np.array_equal(result, np.zeros_like(signal))


class TestBaselineFitSinhFall:
    def test_detects_falling_sinh_segment(self):
        t = np.linspace(0.0, 6.0, 180)
        signal = np.full_like(t, 1.0)
        idx = t >= 2.5
        signal[idx] = -0.12 * np.sinh(0.8 * (t[idx] - 2.5)) + 1.2
        result, ok = baseline_fit_sinh_fall(signal, t)
        assert ok is True
        assert np.any(result > 0.0)

    def test_bad_alignment_is_rejected(self):
        signal = np.array([1.0, 0.8, 0.6])
        t = np.array([0.0, 1.0])
        result, ok = baseline_fit_sinh_fall(signal, t)
        assert ok is False
        assert np.array_equal(result, np.zeros_like(signal))


class TestBaselinePadExponential:
    def test_exponential_padding_decays_to_edge(self):
        onsets = np.array([1.0])
        t = np.array([1.0])
        anchor = np.arange(0.0, 6.0)
        result, ok = baseline_pad_exponential(onsets, t, anchor, width=2.0)
        assert ok is True
        peak_idx = np.searchsorted(anchor, t[0])
        assert result[peak_idx] > 0.0
        assert result[peak_idx] > result[peak_idx + 1] > result[peak_idx + 2]

    def test_negative_width_pads_before_onset(self):
        onsets = np.array([2.0])
        t = np.array([3.0])
        anchor = np.arange(0.0, 7.0)
        result, ok = baseline_pad_exponential(onsets, t, anchor, width=-2.0)
        assert ok is True
        assert np.any(result > 0.0)
        assert result[1] < result[2] < result[3]


class TestBaselinePadGaussian:
    def test_gaussian_padding_has_centered_peak(self):
        onsets = np.array([1.0])
        t = np.array([2.0])
        anchor = np.arange(0.0, 8.0)
        result, ok = baseline_pad_gaussian(onsets, t, anchor, width=2.0)
        assert ok is True
        assert result.shape == anchor.shape
        assert np.argmax(result) >= np.searchsorted(anchor, t[0])
        assert result.max() > result.min()


class TestBaselineCombineProduct:
    def test_elementwise_product_matches_numpy(self):
        a = np.array([1.0, 2.0, 3.0])
        b = np.array([0.5, 1.0, 2.0])
        result, ok = baseline_combine_product([a, b])
        assert ok is True
        assert np.allclose(result, np.array([0.5, 2.0, 6.0]))

    def test_plus_one_variant_avoids_zero_collapse(self):
        a = np.array([0.0, 0.5, 0.0])
        b = np.array([0.2, 0.2, 0.2])
        result, ok = baseline_combine_product([a, b], plus_one=True)
        assert ok is True
        assert np.allclose(result, np.array([0.2, 0.8, 0.2]))


class TestBaselineCombineConvolve:
    def test_convolution_normalizes_each_step(self):
        a = np.array([1.0, 0.0, 0.0])
        b = np.array([0.0, 1.0, 0.0])
        result, ok = baseline_combine_convolve([a, b])
        assert ok is True
        assert result.shape == a.shape
        assert np.isclose(result.sum(), 1.0)


class TestBaselineCombineWeighted:
    def test_weighted_product_uses_weights(self):
        a = np.array([0.5, 0.5, 0.5])
        b = np.array([0.2, 0.2, 0.2])
        result, ok = baseline_combine_weighted([a, b], weights=np.array([2.0, 1.0]))
        assert ok is True
        assert np.allclose(result, np.array([0.2, 0.2, 0.2]))


class TestBaselineCombineCoherence:
    def test_identical_signals_remain_unchanged(self):
        signal = np.sin(np.linspace(0.0, 4.0 * np.pi, 200))
        result, ok = baseline_combine_coherence([signal, signal.copy()])
        assert ok is True
        assert np.allclose(result, signal, atol=1e-2)

    def test_single_component_is_pass_through(self):
        signal = np.array([0.0, 1.0, 0.5, 0.0])
        result, ok = baseline_combine_coherence([signal])
        assert ok is True
        assert np.array_equal(result, signal)
        assert result is not signal

    def test_different_components_still_return_finite_signal(self):
        a = np.sin(np.linspace(0.0, 2.0 * np.pi, 128))
        b = np.cos(np.linspace(0.0, 2.0 * np.pi, 128))
        result, ok = baseline_combine_coherence([a, b], dt=0.5)
        assert ok is True
        assert result.shape == a.shape
        assert np.all(np.isfinite(result))

    def test_three_component_reduction_is_stable(self):
        a = np.sin(np.linspace(0.0, 2.0 * np.pi, 64))
        b = np.sin(np.linspace(0.0, 2.0 * np.pi, 64) + 0.2)
        c = np.sin(np.linspace(0.0, 2.0 * np.pi, 64) - 0.2)
        result, ok = baseline_combine_coherence([a, b, c])
        assert ok is True
        assert result.shape == a.shape
        assert np.all(np.isfinite(result))

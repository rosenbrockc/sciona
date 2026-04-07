"""Tests for baseline-path scoring atoms, registry declarations, and catalog seeding."""

from __future__ import annotations

import numpy as np
import pytest

from sciona.architect.catalog import PrimitiveCatalog, seed_builtin_primitives
from sciona.expansion_atoms.baseline_scoring_registry import (
    BASELINE_SCORING_DECLARATIONS,
    next_baseline_scoring_variant,
)
from sciona.expansion_atoms.runtime_baseline_scoring import (
    accumulate_analyzed_time,
    accumulate_prediction_window_time,
    apply_bmi_correction,
    compute_event_rate_per_hour,
    score_baseline_path,
    score_bmi_baseline_path,
    score_pat_baseline_path,
)


class TestAccumulateAnalyzedTime:
    def test_accumulates_only_contiguous_sleep_regions(self):
        anchor = np.array([0.0, 10.0, 20.0, 30.0, 40.0])
        sleep_mask = np.array([False, True, True, False, True])

        hours, ok = accumulate_analyzed_time(anchor, sleep_mask, seconds_per_hour=1.0)

        assert ok is True
        assert hours == pytest.approx(10.0)

    def test_empty_sleep_mask_returns_zero(self):
        anchor = np.array([0.0, 10.0, 20.0])
        sleep_mask = np.array([False, False, False])

        hours, ok = accumulate_analyzed_time(anchor, sleep_mask)

        assert ok is False
        assert hours == 0.0


class TestAccumulatePredictionWindowTime:
    def test_merges_overlapping_padded_prediction_windows(self):
        probabilities = np.array([0.0, 0.8, 0.0, 0.9, 0.0])
        anchor = np.array([0.0, 10.0, 20.0, 30.0, 40.0])

        hours, ok = accumulate_prediction_window_time(
            probabilities,
            anchor,
            threshold=0.5,
            pad=15.0,
            seconds_per_hour=1.0,
        )

        assert ok is True
        assert hours == pytest.approx(50.0)

    def test_no_positive_probability_returns_zero(self):
        hours, ok = accumulate_prediction_window_time(
            np.zeros(4),
            np.arange(4.0),
        )

        assert ok is False
        assert hours == 0.0


class TestComputeEventRatePerHour:
    def test_counts_unique_region_labels(self):
        regions = np.array([0, 1, 1, 0, 2, 2, 0, 3])

        rate, ok = compute_event_rate_per_hour(regions, 2.0)

        assert ok is True
        assert rate == pytest.approx(1.5)

    def test_supports_primary_path_half_event_scaling(self):
        regions = np.array([0, 1, 1, 0, 2, 2, 0, 3])

        rate, ok = compute_event_rate_per_hour(regions, 2.0, divisor=2.0)

        assert ok is True
        assert rate == pytest.approx(0.75)


class TestApplyBmiCorrection:
    def test_applies_mild_branch_bmi_formula(self):
        corrected, ok = apply_bmi_correction(10.0, 31.0)

        assert ok is True
        assert corrected == pytest.approx(6.9)

    def test_missing_bmi_uses_default(self):
        corrected, ok = apply_bmi_correction(10.0, None)

        assert ok is True
        assert corrected == pytest.approx(6.085207100591715)


class TestBaselinePathScoring:
    def test_score_baseline_path_uses_combined_branch_after_short_night_adjustment(self):
        predictor_regions = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4])
        combined_regions = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6])

        score, ok = score_baseline_path(
            predictor_regions,
            combined_regions,
            analyzed_time_hours=2.0,
            density_hours=4.0,
            moderate_or_severe=False,
        )

        assert ok is True
        assert score == pytest.approx(11.0)

    def test_score_bmi_baseline_path_applies_combined_branch_bmi_formula(self):
        predictor_regions = np.array(
            [0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6, 0, 7, 7]
        )
        combined_regions = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6])

        score, ok = score_bmi_baseline_path(
            predictor_regions,
            combined_regions,
            analyzed_time_hours=2.0,
            density_hours=4.0,
            bmi=31.0,
            moderate_or_severe=False,
        )

        assert ok is True
        assert score == pytest.approx(11.99)


class TestPatBaselinePathScoring:
    def test_pat_baseline_path_uses_short_night_adjustment(self):
        pat_regions = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 0, 4, 4, 0, 5, 5, 0, 6, 6])

        score, ok = score_pat_baseline_path(
            pat_regions,
            analyzed_time_hours=2.0,
            density_hours=4.0,
        )

        assert ok is True
        assert score == pytest.approx(8.0)

    def test_pat_baseline_path_filters_sparse_low_ahi(self):
        pat_regions = np.array([0, 1, 1, 0, 2, 2, 0, 3, 3, 0])

        score, ok = score_pat_baseline_path(
            pat_regions,
            analyzed_time_hours=2.0,
            density_hours=2.5,
        )

        assert ok is False
        assert np.isnan(score)


class TestBaselineScoringRegistry:
    def test_expected_declarations_are_present(self):
        assert set(BASELINE_SCORING_DECLARATIONS) == {
            "accumulate_analyzed_time",
            "accumulate_prediction_window_time",
            "compute_event_rate_per_hour",
            "apply_bmi_correction",
            "score_baseline_path",
            "score_bmi_baseline_path",
            "score_pat_baseline_path",
        }

    def test_no_curated_variants_exist_yet(self):
        assert next_baseline_scoring_variant("score_pat_baseline_path") is None


class TestBaselineScoringCatalogSeeding:
    def test_seed_builtin_primitives_registers_baseline_scoring_aliases(self):
        catalog = PrimitiveCatalog()
        seed_builtin_primitives(catalog)

        analyzed = catalog.get("compute analyzed sleep time")
        density = catalog.get("compute density windows")
        pat = catalog.get("PAT Baseline Score")

        assert analyzed is not None
        assert analyzed.name == "accumulate_analyzed_time"
        assert density is not None
        assert density.name == "accumulate_prediction_window_time"
        assert pat is not None
        assert pat.name == "score_pat_baseline_path"

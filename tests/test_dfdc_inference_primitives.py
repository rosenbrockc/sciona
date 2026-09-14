"""Independent synthetic checks for source threshold and spatial conventions."""

import cv2
import numpy as np
import pytest

from sciona.dfdc_inference_primitives import (
    confident_strategy, isotropically_resize_image, put_to_center,
)


@pytest.mark.parametrize("count,expected", [(11, (11 * .9 + 9 * .3) / 20), (12, .9)])
def test_fake_count_must_exceed_eleven(count, expected):
    values = [.9] * count + [.3] * (20 - count)
    assert confident_strategy(values) == pytest.approx(expected)


@pytest.mark.parametrize("count", [12, 13])
def test_fake_fraction_uses_strict_floor_division(count):
    values = [.9] * count + [.3] * (32 - count)
    expected = .9 if count == 13 else (count * .9 + (32 - count) * .3) / 32
    assert confident_strategy(values) == pytest.approx(expected)


@pytest.mark.parametrize("count,expected", [(9, .17), (10, .1)])
def test_real_fraction_must_exceed_ninety_percent(count, expected):
    assert confident_strategy([.1] * count + [.8] * (10 - count)) == pytest.approx(expected)


def test_threshold_equalities_are_not_confident():
    assert confident_strategy([.8] * 20 + [.2] * 12) == pytest.approx(.575)


def test_aggregation_order_changes_video_score():
    first = np.array([.9] * 13 + [.3] * 19)
    second = np.full(32, .1)
    source_order = np.mean([confident_strategy(first), confident_strategy(second)])
    assert source_order == pytest.approx(.5)
    assert confident_strategy((first + second) / 2) == pytest.approx(.321875)


def test_center_padding_odd_remainder_goes_bottom_right():
    img = np.full((2, 3, 3), 73, dtype=np.uint8)
    expected = np.zeros((5, 5, 3), dtype=np.uint8)
    expected[1:3, 1:4] = 73
    np.testing.assert_array_equal(put_to_center(img, 5), expected)


def test_center_crop_is_top_left_before_padding():
    img = np.arange(6 * 7 * 3, dtype=np.uint8).reshape(6, 7, 3)
    np.testing.assert_array_equal(put_to_center(img, 4), img[:4, :4])


@pytest.mark.parametrize("shape,target,expected_shape", [
    ((3, 7, 3), 5, (2, 5, 3)), ((7, 3, 3), 5, (5, 2, 3)),
    ((2, 3, 3), 8, (5, 8, 3)), ((3, 2, 3), 8, (8, 5, 3)),
])
def test_resize_preserves_aspect_with_truncation(shape, target, expected_shape):
    img = np.full(shape, 91, dtype=np.uint8)
    result = isotropically_resize_image(img, target)
    assert result.shape == expected_shape
    np.testing.assert_array_equal(result, np.full(expected_shape, 91, dtype=np.uint8))


def test_resize_nearest_has_independent_pixel_oracle():
    img = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    got = isotropically_resize_image(img, 6, interpolation_up=cv2.INTER_NEAREST)
    np.testing.assert_array_equal(got, np.repeat(np.repeat(img, 2, axis=0), 2, axis=1))


def test_matching_long_edge_returns_original_object():
    img = np.zeros((3, 7, 3), dtype=np.uint8)
    assert isotropically_resize_image(img, 7) is img

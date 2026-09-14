"""Synthetic replay and independent pixel checks for the explicit reference."""
import numpy as np
import pytest

from sciona.aptos_augmentation import Parameters, apply_parameters, draw_parameters


def synthetic():
    return (np.arange(13 * 19 * 3).reshape(13, 19, 3) % 256).astype(np.uint8)


def test_identity_and_independent_mirror_oracle():
    x = synthetic()
    np.testing.assert_array_equal(apply_parameters(x, Parameters()), x)
    np.testing.assert_array_equal(apply_parameters(x, Parameters(mirror_x=True)), x[:, ::-1])
    np.testing.assert_array_equal(apply_parameters(x, Parameters(mirror_y=True)), x[::-1])


def test_additive_brightness_clipping_oracle():
    x = synthetic()
    expected = np.clip(x.astype(np.int16) + 20, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(apply_parameters(x, Parameters(brightness=20.)), expected)


def test_integer_translation_and_black_fill():
    x = synthetic()
    expected = np.zeros_like(x)
    expected[:, 2:] = x[:, :-2]
    np.testing.assert_array_equal(apply_parameters(x, Parameters(shift_x=2 / 19)), expected)


def test_seeded_replay_and_no_input_mutation():
    x = synthetic()
    before = x.copy()
    first, second = np.random.default_rng(71), np.random.default_rng(71)
    for _ in range(20):
        a, b = draw_parameters(first), draw_parameters(second)
        assert a == b
        output = apply_parameters(x, a)
        np.testing.assert_array_equal(output, apply_parameters(x, b))
        assert output.shape == x.shape and output.dtype == np.uint8
    np.testing.assert_array_equal(x, before)


@pytest.mark.parametrize('parameters', [Parameters(scale=0), Parameters(hue_degrees=11),
    Parameters(brightness=float('nan')), Parameters(filter='unknown'), Parameters(mirror_x=1)])
def test_invalid_parameters_rejected(parameters):
    with pytest.raises(ValueError):
        apply_parameters(synthetic(), parameters)

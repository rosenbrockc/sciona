import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.feng_fft import feng_fft_features


def test_band_means_order_and_standard_deviation():
    x = np.random.default_rng(610).normal(size=(24000, 2)).astype(np.float32)
    before = x.copy()
    actual = feng_fft_features(x)
    assert actual.shape == (2, 7, 2) and actual.dtype == np.float64
    edges = [.1, 4, 8, 12, 30, 70, 180]
    frequencies = np.fft.rfftfreq(12000, 1/400)
    for channel in range(2):
        for frame in range(2):
            w = x[frame*12000:(frame+1)*12000, channel]
            logs = np.log10(np.abs(np.fft.rfft(w)))
            expected = [np.mean(logs[(frequencies >= lo) & (frequencies < hi)], dtype=np.float64) for lo, hi in zip(edges[:-1], edges[1:])]
            np.testing.assert_allclose(actual[channel, :6, frame], expected, rtol=1e-7)
            assert actual[channel, 6, frame] == np.std(w)
    np.testing.assert_array_equal(x, before)


def test_trailing_samples_dropped_and_channels_independent():
    x = np.random.default_rng(611).normal(size=(12000, 2))
    expected = feng_fft_features(x)
    np.testing.assert_array_equal(feng_fft_features(np.r_[x, np.zeros((17, 2))]), expected)
    np.testing.assert_array_equal(feng_fft_features(x[:, ::-1]), expected[::-1])


def test_zero_spectrum_keeps_source_nonfinite_features_for_classifier_cleanup():
    result = feng_fft_features(np.zeros((12000, 1)))
    assert np.all(np.isneginf(result[:, :6]))
    assert result[0, 6, 0] == 0


@pytest.mark.parametrize('x', [np.ones(12000), np.ones((11999, 2)), np.ones((12000, 0)), np.full((12000, 1), np.nan), np.ones((12000, 1), dtype=complex)])
def test_invalid_inputs(x):
    with pytest.raises(ValueError):
        feng_fft_features(x)

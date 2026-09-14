import numpy as np
import pytest
from sciona.atoms.riemannian_bci.covariance_features.relative_power import relative_log_band_power


BANDS = [[0., 4.], [4., 12.], [12., 16.]]


def test_matches_independent_scalar_welch_and_unequal_band_means():
    a = np.random.default_rng(65).normal(size=(2, 3, 95))
    before = a.copy()
    actual = relative_log_band_power(a, BANDS, 32., 32, .25)
    taper = .5 - .5 * np.cos(2 * np.pi * np.arange(32) / 32)
    for wi in range(2):
        for ci in range(3):
            spectra = []
            for start in range(0, 64, 24):
                frame = a[wi, ci, start:start + 32]
                frame = (frame - frame.mean()) * taper
                dft = np.array([sum(frame[t] * np.exp(-2j * np.pi * k * t / 32) for t in range(32)) for k in range(17)])
                power = abs(dft) ** 2 / (32 * sum(taper ** 2))
                power[1:-1] *= 2
                spectra.append(power)
            spectrum = np.mean(spectra, axis=0)
            means = np.array([spectrum[:4].mean(), spectrum[4:12].mean(), spectrum[12:16].mean()])
            np.testing.assert_allclose(actual[wi, ci], np.log(means / means.sum()), atol=1e-13)
    np.testing.assert_array_equal(a, before)
    np.testing.assert_allclose(np.exp(actual).sum(axis=-1), 1.)
    np.testing.assert_allclose(relative_log_band_power(-7 * a, BANDS, 32., 32, .25), actual, atol=1e-13)


@pytest.mark.parametrize('kwargs', [
    {'frequency_bands': [[.1, .2]]}, {'frequency_bands': [[4., 2.]]},
    {'frequency_bands': []}, {'fs': 0.}, {'overlap': 1.}, {'fft_window': 128},
])
def test_invalid_spectral_domains_rejected(kwargs):
    args = dict(frequency_bands=BANDS, fs=32., fft_window=32, overlap=.25)
    args.update(kwargs)
    with pytest.raises(ValueError):
        relative_log_band_power(np.random.default_rng(1).normal(size=(1, 2, 64)), **args)


def test_constant_and_nonfinite_channels_rejected():
    for value in [0., 3., np.nan, np.inf]:
        with pytest.raises(ValueError):
            relative_log_band_power(np.full((1, 2, 64), value), BANDS, 32., 32)

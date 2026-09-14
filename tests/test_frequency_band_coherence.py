import numpy as np
import pytest
from sciona.atoms.riemannian_bci.covariance_features.frequency_coherence import frequency_band_coherence


BANDS = [[0., 4.], [4., 8.], [8., 16.]]


def test_coherence_matches_scalar_dft_reference_and_axis_layout():
    a = np.random.default_rng(51).normal(size=(2, 3, 83))
    before = a.copy()
    actual = frequency_band_coherence(a, BANDS, fs=32., fft_window=16, overlap=.5)
    assert actual.shape == (2, 3, 3, 3)
    for wi in range(2):
        for ci in range(3):
            coefficients = []
            for start in range(0, 68, 8):
                frame = a[wi, ci, start:start + 16] * np.hanning(16)
                dft = [sum(frame[t] * np.exp(-2j * np.pi * k * t / 16) for t in range(16)) for k in range(8)]
                coefficients.append([np.mean(dft[:2]), np.mean(dft[2:4]), np.mean(dft[4:8])])
            z = np.array(coefficients)
            for i in range(3):
                for j in range(3):
                    expected = abs(sum(z[:, i] * z[:, j].conj())) ** 2 / (sum(abs(z[:, i]) ** 2) * sum(abs(z[:, j]) ** 2))
                    assert actual[wi, i, j, ci] == pytest.approx(expected, abs=1e-13)
            assert np.linalg.eigvalsh(actual[wi, :, :, ci]).min() >= -1e-12
    np.testing.assert_array_equal(a, before)
    np.testing.assert_allclose(frequency_band_coherence(a * -7, BANDS, 32., 16, .5), actual, atol=1e-14)


def test_one_frame_is_singular_and_not_implicitly_regularized():
    a = np.random.default_rng(2).normal(size=(1, 1, 16))
    matrix = frequency_band_coherence(a, BANDS, 32., 16)[0, :, :, 0]
    np.testing.assert_allclose(matrix, np.ones((3, 3)))
    assert np.linalg.matrix_rank(matrix) == 1


@pytest.mark.parametrize('kwargs', [
    {'overlap': 1.}, {'overlap': -.1}, {'fs': 0.}, {'fft_window': 2},
    {'fft_window': 256}, {'frequency_bands': [[.1, .2]]},
    {'frequency_bands': [[4., 2.]]}, {'frequency_bands': [[0., 17.]]},
])
def test_invalid_spectral_contract_rejected(kwargs):
    args = dict(frequency_bands=BANDS, fs=32., fft_window=16, overlap=.5)
    args.update(kwargs)
    with pytest.raises(ValueError):
        frequency_band_coherence(np.ones((1, 2, 40)), **args)


def test_zero_power_rejected():
    with pytest.raises(ValueError, match='power'):
        frequency_band_coherence(np.zeros((1, 2, 40)), BANDS, 32., 16)

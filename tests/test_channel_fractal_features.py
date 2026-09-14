import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.fractal_features import channel_fractal_features


def test_linear_ramp_exposes_source_hfd_normalization():
    result = channel_fractal_features(np.arange(10.)[None, None])[0, 0]
    assert result[0] == pytest.approx(1.)
    assert result[1] == pytest.approx(0., abs=1e-14)
    assert np.isfinite(result[2])


def test_cyclic_pfd_and_hurst_match_scalar_reference_without_mutation():
    values = np.array([0., 1., 3., 2., 5., 4., 4., 8.])
    before = values.copy()
    result = channel_fractal_features(values[None, None])[0, 0]
    signs = [np.sign(values[i+1] - values[i]) for i in range(len(values)-1)]
    changes = sum(signs[i] != signs[i-1] for i in range(len(signs)))
    expected_pfd = np.log10(8) / (np.log10(8) + np.log10(8/(8+.4*changes)))
    assert result[0] == pytest.approx(expected_pfd)
    centered = values - sum(values)/len(values)
    cumulative = [sum(centered[:i+1]) for i in range(len(values))]
    dependent = []
    for length in range(2, len(values)+1):
        subset = centered[:length]
        mean = sum(subset)/length
        sd = (sum((v-mean)**2 for v in subset)/(length-1))**.5
        dependent.append(np.log((max(cumulative[:length])-min(cumulative[:length])+1e-12)/(sd or 1e-12)))
    x = np.log(np.arange(1., len(values)))
    y = np.array(dependent)
    slope = sum((x-x.mean())*(y-y.mean()))/sum((x-x.mean())**2)
    assert result[2] == pytest.approx(slope, abs=1e-13)
    np.testing.assert_array_equal(values, before)


@pytest.mark.parametrize('values', [np.ones(20), np.tile([0., 1.], 10), np.arange(3.), np.full(10, np.nan)])
def test_invalid_feature_regimes_rejected(values):
    with pytest.raises(ValueError):
        channel_fractal_features(values[None, None])

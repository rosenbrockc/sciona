import numpy as np
import pytest

from sciona.contrails_prediction import VARIANTS, encode_mask, ensemble_probabilities


@pytest.mark.parametrize('variant', ['v43', 'v47'])
def test_unnormalized_weights_change_threshold_decision(variant):
    (wt, nt), (ws, ns) = VARIANTS[variant]
    temporal = np.full((nt, 1, 1, 256, 256), .48, np.float32)
    single = np.full((ns, 1, 1, 256, 256), .48, np.float32)
    result = ensemble_probabilities(temporal, single, variant=variant)
    np.testing.assert_allclose(result, .48 * (wt + ws), rtol=1e-6)
    assert np.all(result > .5)
    assert encode_mask(result[0]) == '1 65536'


def test_rle_empty_strict_threshold_and_column_order():
    x = np.zeros((256, 256), np.float32)
    x[0, 0] = .5
    assert encode_mask(x) == '-'
    x[1:3, 0] = 1
    x[0, 1] = 1
    assert encode_mask(x) == '2 2 257 1'


def test_missing_fold_and_nonprobability_rejected():
    valid = np.zeros((5, 1, 1, 256, 256), np.float32)
    with pytest.raises(ValueError):
        ensemble_probabilities(valid[:4], valid)
    invalid = valid.copy()
    invalid[0, 0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        ensemble_probabilities(invalid, valid)

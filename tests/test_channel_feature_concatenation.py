import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.feature_concatenation import concatenate_channel_features


def test_family_concatenation_precedes_channel_flattening():
    blocks = [np.array([[[base, base+1], [base+10, base+11]]]) for base in [0., 100., 200., 300.]]
    result = concatenate_channel_features(*blocks)
    assert result.shape == (1, 2, 8)
    np.testing.assert_array_equal(result.reshape(1, -1)[0],
        [0, 1, 100, 101, 200, 201, 300, 301, 10, 11, 110, 111, 210, 211, 310, 311])
    result[:] = -1
    assert all(np.all(b >= 0) for b in blocks)


@pytest.mark.parametrize('bad', [np.ones((1, 3, 2)), np.ones((2, 2, 2)), np.ones((1, 2, 0)), np.full((1, 2, 2), np.nan)])
def test_incompatible_feature_blocks_rejected(bad):
    good = np.ones((1, 2, 2))
    with pytest.raises(ValueError):
        concatenate_channel_features(good, bad, good, good)

import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.feng_partitions import feng_filter_partitions, feng_feature_partitions


@pytest.mark.parametrize('function', [feng_filter_partitions, feng_feature_partitions])
@pytest.mark.parametrize('parts', [([], [np.ones((10, 2))]), ([np.ones((10, 2))], []),
                                 ([np.ones((10, 2))], [np.ones((10, 3))]),
                                 ([np.ones(10)], [np.ones((10, 2))])])
def test_partition_boundaries_reject_empty_or_mismatched_inputs(function, parts):
    with pytest.raises(ValueError):
        function(*parts)


def test_feature_partition_requires_exact_source_duration():
    with pytest.raises(ValueError):
        feng_feature_partitions([np.zeros((12000, 2))], [np.zeros((240000, 2))])


def test_source_duration_zero_features_preserve_partition_order_and_shape():
    training, prediction = feng_feature_partitions([np.zeros((240000, 1))]*2, [np.zeros((240000, 1))])
    assert training.shape == (2, 1, 7, 20) and prediction.shape == (1, 1, 7, 20)
    assert np.all(np.isneginf(training[:, :, :6]))
    assert np.all(training[:, :, 6] == 0)
    assert not np.shares_memory(training, prediction)

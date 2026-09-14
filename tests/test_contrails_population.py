import numpy as np
import pytest

from sciona.contrails_population import FOLDS, select_folds


@pytest.mark.parametrize('size', [10, 23, 101])
@pytest.mark.parametrize('variant', ['v43', 'v47'])
@pytest.mark.parametrize('branch', ['single', 'temporal'])
def test_source_folds_against_independent_permutation(size, variant, branch):
    permutation = np.random.RandomState(42).permutation(size)
    sizes = np.full(10, size // 10)
    sizes[:size % 10] += 1
    chunks = np.split(permutation, np.cumsum(sizes)[:-1])
    folds = select_folds(size, branch=branch, variant=variant)
    assert [fold for fold, _, _ in folds] == list(FOLDS[variant][branch])
    for fold, train, val in folds:
        np.testing.assert_array_equal(val, np.sort(chunks[fold]))
        np.testing.assert_array_equal(train, np.setdiff1d(np.arange(size), chunks[fold]))
        assert not set(train) & set(val)
        assert len(train) + len(val) == size


@pytest.mark.parametrize('size', [0, 9, True, 10.5])
def test_invalid_population_size(size):
    with pytest.raises(ValueError):
        select_folds(size, branch='single')

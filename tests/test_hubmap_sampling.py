import numpy as np
import pytest
from sciona.hubmap_sampling import balanced_epoch_indices


def test_sampling_reproducible_and_inputs_preserved():
    present = np.array([False]*4+[True]*6)
    bins = np.array([0]*4+[1,2,3,4,5,6])
    before = bins.copy()
    a = balanced_epoch_indices(present,bins,maximum_bin=3,rng=np.random.RandomState(9))
    b = balanced_epoch_indices(present,bins,maximum_bin=3,rng=np.random.RandomState(9))
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(bins,before)
    assert np.all(~present[a[-4:]]) and np.all(present[a[:-4]])


@pytest.mark.parametrize('present,bins', [([True,True],[1,2]),([False,False],[0,0]),([True,False],[1.,0.]),([True,False],[-1,0])])
def test_invalid_population_does_not_advance_rng(present,bins):
    rng=np.random.RandomState(7);before=rng.get_state()
    with pytest.raises(ValueError): balanced_epoch_indices(present,bins,maximum_bin=3,rng=rng)
    after=rng.get_state()
    assert before[0]==after[0] and before[2:]==after[2:]
    np.testing.assert_array_equal(before[1],after[1])

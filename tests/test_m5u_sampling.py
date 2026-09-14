import numpy as np
import pytest
from sciona.m5u_sampling import sample,quantile_mask


def test_integral_replication_counts_and_shared_repeat_horizons():
    rows,horizons=sample([1.,3.],[12,12],12,2,repeats=3,seed=4)
    assert np.bincount(rows).tolist()==[3,9]
    np.testing.assert_equal(rows[:4],[0,1,1,1])
    np.testing.assert_equal(horizons[:4],horizons[4:8])
    np.testing.assert_equal(horizons[:4],horizons[8:])
    assert horizons.dtype==np.int8 and ((horizons>=1)&(horizons<=28)).all()


def test_rng_consumption_and_scalar_replication_oracle():
    weights=np.array([1.,2.,6.,50.]);levels=np.array([4,4,4,5]);seed=98;fraction=1.5
    rng=np.random.RandomState(seed);ratio=weights/3*fraction
    draw=rng.rand(4);initial=[i for i in range(4) if levels[i]==4 and ratio[i]>draw[i]]
    result=initial.copy();remaining=[ratio[i] for i in initial]
    while max(remaining)>1:
        remaining=[x-1 for x in remaining];draw=rng.rand(len(initial))
        result.extend(initial[i] for i in range(len(initial)) if remaining[i]>draw[i])
    horizons=rng.randint(0,28,len(result))+1
    actual,actual_horizon=sample(weights,levels,4,fraction,seed=seed)
    np.testing.assert_equal(actual,result);np.testing.assert_equal(actual_horizon,horizons)


def test_quantile_group_exclusion_and_level_exponent():
    groups=np.tile([1,2,3],100);u=np.random.RandomState(9).rand(300)
    for level,power in [(10,.25),(11,.35),(12,.35)]:
        np.testing.assert_equal(quantile_mask(groups,2,.01,level,seed=9),(groups!=2)&(u<.01**power))


@pytest.mark.parametrize('weights',[[0.,0.],[-1.,2.],[np.nan,1.]])
def test_invalid_sampling_population_rejects(weights):
    with pytest.raises(ValueError):sample(weights,[1,1],1,1.)

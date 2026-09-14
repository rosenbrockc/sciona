"""Synthetic independent threshold and random fold oracles."""
import numpy as np
import pytest
from sklearn.metrics import matthews_corrcoef
from sciona.vsb_threshold import threshold,repeated_folds


def test_threshold_bruteforce_ties_and_strict_application():
    rng=np.random.default_rng(97)
    for _ in range(100):
        y=rng.integers(0,2,40);p=rng.integers(0,8,40)/8
        candidates=np.unique(p);scores=[matthews_corrcoef(y,p>=v) for v in candidates]
        wanted=max(range(len(scores)),key=lambda i:(scores[i],i))
        result=threshold(y,p)
        assert result['threshold']==candidates[wanted]
        assert result['mcc']==pytest.approx(scores[wanted],abs=1e-14)
    result=threshold([0,1],[.1,.9]);assert result['threshold']==.9
    assert (.9>result['threshold']) is False


def test_constant_and_one_class_thresholds():
    assert threshold([0,1],[.5,.5])==dict(threshold=.5,mcc=0.)
    assert threshold([1,1],[.1,.9])==dict(threshold=.9,mcc=0.)


def test_legacy_partition_and_multiplicity_without_rng_mutation():
    y=np.tile([0,1],100);state=np.random.get_state()
    plans=repeated_folds(y,seed=123948,repetitions=2)
    after=np.random.get_state();np.testing.assert_array_equal(state[1],after[1]);assert state[2:]==after[2:]
    for iteration in range(2):
        rng=np.random.RandomState(123948+iteration);assignment=np.zeros(len(y),int)
        assignment[y==1]=rng.randint(0,5,(y==1).sum());assignment[y==0]=rng.randint(0,5,(y==0).sum())
        counts=np.zeros((3,len(y)),int)
        for fold,parts in enumerate(plans[iteration*5:iteration*5+5]):
            np.testing.assert_array_equal(parts[1],np.flatnonzero(assignment==fold))
            np.testing.assert_array_equal(parts[2],np.flatnonzero(assignment==(fold+1)%5))
            assert len(set(parts[0])&set(parts[1]))==0
            for index,part in enumerate(parts):counts[index,part]+=1
        np.testing.assert_array_equal(counts,np.broadcast_to(np.array([3,1,1])[:,None],counts.shape))


def test_invalid_inputs():
    with pytest.raises(ValueError):threshold([True,False],[.1,.2])
    with pytest.raises(ValueError):threshold([0,1],[.1,float('nan')])
    with pytest.raises(ValueError):repeated_folds([0,1],seed=1,repetitions=1)

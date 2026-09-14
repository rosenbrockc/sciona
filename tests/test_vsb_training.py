"""Actual native five-fold ensemble and signal-label threshold routing."""
import numpy as np
from sciona.vsb_training import fit
from sciona.vsb_threshold import threshold


def test_actual_full_repetition_count_and_threshold_routing():
    rng=np.random.default_rng(98);x=rng.normal(size=(200,9));labels=np.zeros((200,3),int)
    labels[x[:,0]>0,0]=1
    x[0,2]=np.nan;q=rng.normal(size=(8,9))
    result=fit(x,labels,q,rounds=3,patience=2)
    assert result['models']==125 and len(result['best_iterations'])==125
    assert all(1<=n<=3 for n in result['best_iterations'])
    expected=threshold(labels.reshape(-1),np.repeat(result['training_probabilities'],3))
    assert result['threshold']==expected['threshold']
    assert result['probabilities'].shape==(8,) and np.isfinite(result['probabilities']).all()
    np.testing.assert_array_equal(result['signal_decisions'],np.repeat((result['probabilities']>result['threshold'])[:,None],3,axis=1))
    for key in ('training_probabilities','validation_probabilities','test_probabilities'):
        assert np.all((result[key]>=0)&(result[key]<=1))

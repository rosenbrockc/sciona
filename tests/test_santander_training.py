"""Complete synthetic neural branch lifecycle and scheduler checks."""
import numpy as np
import pytest
from scipy.stats import rankdata
from sciona.santander_training import train_neural_cv, cycle_parameters


def sample():
    rng=np.random.default_rng(31)
    return rng.integers(0,9,(20,4)).astype(float),np.arange(20)%2,rng.integers(0,12,(6,4)).astype(float)


def test_full_fold_seed_training_checkpoint_restore_and_rank_mean():
    controls=dict(folds=2,seeds=(42,7),epochs=4,batch_size=8)
    first=train_neural_cv(*sample(),**controls)
    second=train_neural_cv(*sample(),**controls)
    np.testing.assert_array_equal(first['model_probabilities'],second['model_probabilities'])
    assert len(first['models'])==4
    assert {(r['fold'],r['seed']) for r in first['models']}=={(0,42),(0,7),(1,42),(1,7)}
    for r in first['models']:
        assert r['optimizer_steps']==12 and r['fit_rows']==r['validation_rows']==10
        assert r['restored_auc']==max(r['validation_auc'])
        assert r['best_epoch']==r['validation_auc'].index(max(r['validation_auc']))
    expected=np.mean([rankdata(p) for p in first['model_probabilities']],axis=0)
    np.testing.assert_array_equal(first['mean_ranks'],expected)
    assert ((first['model_probabilities']>=0)&(first['model_probabilities']<=1)).all()


def test_legacy_schedule_phase_values():
    assert cycle_parameters(0,10,.01)==(.0004,.95)
    lr,beta=cycle_parameters(3,10,.01)
    assert lr==pytest.approx(.01) and beta==pytest.approx(.85)
    lr,beta=cycle_parameters(9,10,.01)
    assert 0<lr<.001 and .94<beta<.95


@pytest.mark.parametrize('controls',[{'folds':11},{'epochs':0},{'seeds':(1,1)},{'batch_size':1},{'maximum_lr':float('nan')}])
def test_invalid_controls(controls):
    with pytest.raises(ValueError):train_neural_cv(*sample(),**controls)


def test_singleton_tail_preserves_each_scheduled_row():
    from sciona.santander_training import _batches
    order=np.arange(9)
    batches=_batches(order,4)
    assert [len(b) for b in batches]==[4,5]
    np.testing.assert_array_equal(np.concatenate(batches),order)

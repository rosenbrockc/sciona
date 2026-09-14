import numpy as np
import pytest
from sciona import otto_classical_models as module

CONFIGS=[('logistic',dict(C=1.,max_iter=500)),('extra_trees',dict(n_estimators=24,max_features=1.,min_samples_leaf=1)),('multinomial_nb',dict(alpha=1.))]


def fixture():
    y=np.repeat(np.arange(9),10);f=np.tile(np.arange(10)%5,9)
    x=np.eye(9)[y]*20+np.random.default_rng(3).uniform(0,1,(90,9))
    return x,y,f,[f'synthetic-t{i}' for i in range(90)],np.eye(9)*20,[f'synthetic-q{i}' for i in range(9)]


@pytest.mark.parametrize('family,controls',CONFIGS)
def test_three_families_complete_oof_and_full_refit(family,controls):
    result=module.crossfit(*fixture(),family=family,controls=controls,seed=12)
    assert result['fitting_runs']==6 and result['oof'].shape==(90,9) and result['query'].shape==(9,9)
    np.testing.assert_allclose(result['oof'].sum(axis=1),1.)
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


@pytest.mark.parametrize('family,controls',CONFIGS)
def test_own_fold_labels_and_query_are_isolated(family,controls):
    values=list(fixture());a=module.crossfit(*values,family=family,controls=controls,seed=12)
    held=values[2]==0;values[1]=values[1].copy();values[1][held]=(values[1][held]+1)%9
    values[4]=values[4]*3
    b=module.crossfit(*values,family=family,controls=controls,seed=12)
    np.testing.assert_array_equal(a['oof'][held],b['oof'][held])


def test_naive_bayes_full_refit_matches_hand_log_likelihood():
    x,y,f,ids,q,qids=fixture()
    result=module.crossfit(x,y,f,ids,q,qids,family='multinomial_nb',controls=dict(alpha=1.),seed=12)
    counts=np.array([np.log1p(x[y==label]).sum(axis=0)+1 for label in range(9)])
    logp=np.log(counts/counts.sum(axis=1,keepdims=True))
    scores=np.log1p(q)@logp.T+np.log(np.full(9,1/9))
    expected=np.exp(scores-scores.max(axis=1,keepdims=True));expected/=expected.sum(axis=1,keepdims=True)
    np.testing.assert_allclose(result['query'],expected)


def test_identity_overlap_rejected_before_model_fit(monkeypatch):
    values=list(fixture());values[5][0]=values[3][0]
    def forbidden(*args,**kwargs):raise AssertionError('Model construction started')
    monkeypatch.setattr(module,'_model',forbidden)
    with pytest.raises(ValueError):module.crossfit(*values,family='multinomial_nb',controls=dict(alpha=1.),seed=12)

"""Actual full-epoch neural folds and best-checkpoint invariants."""
import numpy as np
import torch
from sciona.amex_neural_training import fit


def test_actual_both_variants_full_epochs_and_checkpoint_selection():
    rng=np.random.default_rng(115);labels=[0]*325+[1]*325
    series=[rng.normal(loc=2*y-1,scale=.2,size=(3+i%11,4)) for i,y in enumerate(labels)]
    features=rng.uniform(0,.1,(650,6))+.8*np.asarray(labels)[:,None]
    query=[rng.normal(loc=2*(i%2)-1,scale=.2,size=(3+i,4)) for i in range(8)]
    qf=rng.uniform(0,.1,(8,6))+.8*(np.arange(8)%2)[:,None]
    before=torch.random.get_rng_state().clone();r=fit(series,features,labels,query,qf)
    assert torch.equal(before,torch.random.get_rng_state())
    assert r['models']==10 and r['epochs']==10 and r['hidden_width']==128
    for variant in r['variants'].values():
        assert len(variant['folds'])==5
        for fold in variant['folds']:
            history=fold['history'];assert len(history)==10
            assert [h['learning_rate'] for h in history]==[.001]*5+[.0001]*4+[.00001]
            assert all(h['training_rows']==512 for h in history)
            best=max(h['metric'] for h in history)
            assert fold['selected_epoch']==next(h['epoch'] for h in history if h['metric']==best)
            assert fold['best_metric']==best and best>0
        scores=variant['query_predictions'];assert np.isfinite(scores).all()
        assert np.mean((scores>=.5)==(np.arange(8)%2))==1.

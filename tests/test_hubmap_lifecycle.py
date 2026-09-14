"""Orchestration-only witnesses; full numerical lifecycle has separate evidence."""
import numpy as np
import pytest
import sciona.hubmap_lifecycle as lifecycle


def test_ordered_checkpoint_ensembles_and_pseudo_handoff(monkeypatch):
    stages=[object() for _ in range(4)]
    image=np.zeros((2,2,3),dtype=np.uint8)
    calls=[]
    def train(stage,slides,groups,*,pseudo_sources,tile_size):
        index=stages.index(stage)
        if index<2:assert pseudo_sources is None
        else:
            assert len(pseudo_sources)==2
            for source in pseudo_sources:
                assert source[0][0] is image
                np.testing.assert_array_equal(source[0][1],np.ones((2,2),dtype=np.uint8))
        return dict(epochs=[index],checkpoints={'best_loss':(index,'loss'),'best_score':(index,'score')})
    def infer(models,slide,**options):
        calls.append((list(models),options['tta']))
        return np.ones((2,2),dtype=np.uint8)
    monkeypatch.setattr(lifecycle,'train_fold',train)
    monkeypatch.setattr(lifecycle,'load_inference_model',lambda payload,**kwargs:payload)
    monkeypatch.setattr(lifecycle,'predict_slide',infer)
    result=lifecycle.run_lifecycle([],[],[[image],[image]],[image],initial_stages=stages[:2],
        retraining_stages=stages[2:],initial_selectors=[(1,'best_score'),(0,'best_loss'),(1,'best_score')],
        final_selectors=[(1,'best_loss'),(0,'best_score')],pseudo_rng={},final_rng={})
    assert calls==[([(1,'score'),(0,'loss'),(1,'score')],4)]*2+[([(3,'loss'),(2,'score')],3)]
    assert result['checkpoints']==[(3,'loss'),(2,'score')]
    assert result['initial_epochs']==[[0],[1]]
    assert result['retraining_epochs']==[[2],[3]]


def test_invalid_fold_selector_rejects_before_training(monkeypatch):
    def unexpected(*args,**kwargs):raise AssertionError('Training started before selector validation')
    monkeypatch.setattr(lifecycle,'train_fold',unexpected)
    with pytest.raises(ValueError,match='selectors'):
        lifecycle.run_lifecycle([],[],[],[],initial_stages=[object()],retraining_stages=[object()],
            initial_selectors=[(1,'best_loss')],final_selectors=[(0,'best_loss')],pseudo_rng={},final_rng={})

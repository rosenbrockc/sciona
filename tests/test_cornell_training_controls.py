"""Checkpoint/control tests with synthetic stand-ins, separate from full-model evidence."""
import numpy as np
import pytest
import torch
from sciona.cornell_training import train_phase


class Model(torch.nn.Module):
    def __init__(self):super().__init__();self.value=torch.nn.Parameter(torch.tensor(.1))
    def forward(self,inputs):
        x,_=inputs
        return {'clipwise_output':self.value.sigmoid().expand(len(x),x.shape[1],264)}


class Source:
    def loss(self,*,augmented):
        return lambda pred,target:((pred-target['all_labels']).square().mean(),{})


def train(epoch):
    yield np.zeros((2,1,960000),dtype=np.float32),np.zeros((2,1,264)),np.zeros((2,1,264))


def validation():
    yield np.zeros((1,2,960000),dtype=np.float32),np.zeros((1,2,264)),np.zeros((1,2,264))


def run(directory,**kwargs):
    settings=dict(training_batches=train,validation_batches=validation,steps_per_epoch=1,
        epochs=2,peak=.001,mixup=False,augmented_loss=False,seed=1,checkpoint_directory=directory)
    settings.update(kwargs)
    return train_phase(Source(),Model(),**settings)


def test_ranked_retention_and_explicit_selection(tmp_path,monkeypatch):
    scores=iter([.1,.5,.2,.8,.4,.7,.6])
    monkeypatch.setattr('sciona.cornell_training.clip_f1',lambda *args:next(scores))
    result=run(tmp_path/'best',epochs=14)
    assert result['selected_epoch']==8
    assert result['retained_epochs']==[10,4,14,12,8]
    assert len(list((tmp_path/'best').glob('*.pt')))==5
    scores=iter([.1,.5,.2,.8,.4,.7,.6])
    result=run(tmp_path/'explicit',epochs=14,select_epoch=4)
    assert result['selected_epoch']==4


def test_unretained_epoch_rejected(tmp_path):
    with pytest.raises(ValueError,match='not retained'):run(tmp_path,select_epoch=1)


def test_missing_batches_rejected(tmp_path):
    with pytest.raises(ValueError,match='Incomplete training epoch'):
        run(tmp_path,training_batches=lambda epoch:iter(()))


def test_extra_batches_rejected(tmp_path):
    with pytest.raises(ValueError,match='Too many training batches'):
        run(tmp_path,training_batches=lambda epoch:iter([next(train(epoch)),next(train(epoch))]))


def test_empty_validation_rejected(tmp_path):
    with pytest.raises(ValueError,match='Empty validation population'):
        run(tmp_path,validation_batches=lambda:iter(()))

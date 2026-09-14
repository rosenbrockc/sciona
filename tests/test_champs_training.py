"""Training boundary validation, with synthetic inputs only."""
from dataclasses import replace

import pytest
import torch

from sciona.champs_training import ChampsTrainer,TrainingOptions


@pytest.mark.parametrize('change',[
    {'optim':'unknown'},{'scheduler':'unknown'},{'batch_size':0},
    {'batch_size':3,'batch_chunk':2},{'batch_size':4,'batch_chunk':2,'champs_loss':True},
    {'seed':True},{'seed':2**32},{'warmup_step':-1},{'max_step':False},
    {'lr':float('nan')},{'lr':0},{'cutout':1.1},{'clip':0},
    {'max_bond_count':407},{'eta_min':-1},
])
def test_invalid_options_fail(change):
    with pytest.raises(ValueError):
        replace(TrainingOptions(),**change).validate()


def test_partial_batch_rejected_before_model_execution():
    trainer=object.__new__(ChampsTrainer)
    trainer.options=TrainingOptions(batch_size=4)
    with pytest.raises(ValueError,match='full consistent batches'):
        trainer.epoch([tuple(torch.zeros(3,1) for _ in range(10))])


def test_empty_epoch_rejected():
    trainer=object.__new__(ChampsTrainer)
    with pytest.raises(ValueError,match='at least one'):
        trainer.epoch([])

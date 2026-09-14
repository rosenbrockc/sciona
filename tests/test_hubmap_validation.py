import pytest
import torch
from sciona.hubmap_validation import ValidationAccumulator
from sciona.hubmap_checkpoint_policy import CheckpointPolicy


def test_zero_dice_denominator_rejects():
    a=ValidationAccumulator()
    a.add(torch.zeros(1,1,2,2),torch.zeros(1,1,2,2))
    with pytest.raises(ValueError,match='undefined'):a.finish(1)


def test_missing_validation_examples_rejects():
    a=ValidationAccumulator()
    a.add(torch.ones(1,1,2,2),torch.ones(1,1,2,2))
    with pytest.raises(ValueError,match='Complete'):a.finish(2)


def test_stop_precedes_new_score_and_snapshot():
    policy=CheckpointPolicy()
    assert policy.decide(epoch=1,validation_loss=1.,validation_score=.5,early_stopping=True,patience=0)['save']==['best_loss','best_score']
    result=policy.decide(epoch=18,validation_loss=1.,validation_score=.9,early_stopping=True,patience=0)
    assert result==dict(save=[],stop=True,step_scheduler=False)
    assert policy.best_score==.5


def test_disabled_early_stop_does_not_update_best_loss_tracking():
    policy=CheckpointPolicy()
    result=policy.decide(epoch=19,validation_loss=1.,validation_score=.5,early_stopping=False,patience=0)
    assert result['save']==['best_loss','best_score','snapshot']
    assert policy.best_loss==1e99 and policy.best_epoch==0

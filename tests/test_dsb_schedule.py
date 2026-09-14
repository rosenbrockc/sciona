import pytest
import torch
from sciona.dsb_schedule import learning_rate, epoch_steps, training_mode


def test_source_strict_schedule_boundaries_and_terminal_epoch():
    assert learning_rate(50,(50,100),(.1,.01)) == .1
    assert learning_rate(51,(50,100),(.1,.01)) == .01
    assert learning_rate(100,(50,100),(.1,.01)) == .01
    with pytest.raises(ValueError): learning_rate(101,(50,100),(.1,.01))


def test_first_epoch_debug_and_alternating_boundary():
    steps=epoch_steps(30,30)
    assert [s['task'] for s in steps] == ['classifier','detector','classifier']
    assert steps[0] == dict(task='classifier',learning_rate=0.,max_batches=5)
    assert [s['task'] for s in epoch_steps(20,1)] == ['detector']
    assert [s['task'] for s in epoch_steps(160,121,4)] == ['classifier']


def test_frozen_statistics_leave_affine_gradients_and_dropout_training():
    model=torch.nn.Sequential(torch.nn.BatchNorm3d(2),torch.nn.Dropout(.5))
    training_mode(model,True)
    before=model[0].running_mean.clone()
    model(torch.ones(2,2,2,2,2)).sum().backward()
    torch.testing.assert_close(model[0].running_mean,before)
    assert model[0].weight.requires_grad and model[0].weight.grad is not None
    assert model[1].training

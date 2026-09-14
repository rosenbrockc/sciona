import pytest
import torch

from sciona.contrails_checkpoints import checked_load, initialize_model, terminal_checkpoint


def test_invalid_late_key_cannot_partially_modify_model():
    model = torch.nn.Sequential(torch.nn.Linear(2, 3), torch.nn.Linear(3, 1))
    before = terminal_checkpoint(model)
    bad = {k: v + 1 for k, v in before.items()}
    bad['1.bias'] = torch.zeros(7)
    with pytest.raises(ValueError):
        checked_load(model, bad)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_terminal_snapshot_cannot_change_after_further_training():
    model = torch.nn.Linear(2, 1)
    snapshot = terminal_checkpoint(model)
    saved = snapshot['weight'].clone()
    with torch.no_grad():
        model.weight.add_(7)
    assert not model.training
    torch.testing.assert_close(snapshot['weight'], saved, rtol=0, atol=0)
    checked_load(model, snapshot)
    torch.testing.assert_close(model.weight, saved, rtol=0, atol=0)


@pytest.mark.parametrize('policy,state', [('unknown', None), ('random', {}), ('encoder', None)])
def test_explicit_checkpoint_policy(policy, state):
    model = torch.nn.Module()
    model.encoder = torch.nn.Linear(1, 1)
    with pytest.raises(ValueError):
        initialize_model(model, policy=policy, state=state)

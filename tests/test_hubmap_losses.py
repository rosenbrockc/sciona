import pytest
import torch
from sciona.hubmap_losses import training_loss


def test_empty_deep_head_is_not_connected_to_gradient():
    logits = torch.zeros(2, 1, 3, 3, requires_grad=True)
    deep = torch.ones_like(logits, requires_grad=True)
    target = torch.zeros_like(logits)
    loss = training_loss(logits, target, deep_logits=[deep])
    loss.backward()
    assert logits.grad is not None and deep.grad is None


def test_deep_loss_only_updates_nonempty_images():
    logits = torch.zeros(2, 1, 3, 3, requires_grad=True)
    deep = torch.ones_like(logits, requires_grad=True)
    target = torch.zeros_like(logits); target[1, 0, 1, 1] = 1
    training_loss(logits, target, deep_logits=[deep]).backward()
    assert torch.count_nonzero(deep.grad[0]) == 0
    assert torch.count_nonzero(deep.grad[1]) > 0


@pytest.mark.parametrize('problem', ['fractional', 'nan', 'shape', 'dtype', 'classification'])
def test_invalid_contract_rejects_before_backward(problem):
    logits = torch.zeros(2, 1, 3, 3, requires_grad=True)
    target = torch.zeros_like(logits)
    args = {}
    if problem == 'fractional': target[0, 0, 0, 0] = .5
    if problem == 'nan': target[0, 0, 0, 0] = float('nan')
    if problem == 'shape': target = target[:, :, :, :2]
    if problem == 'dtype': target = target.double()
    if problem == 'classification': args['classification_logits'] = torch.zeros(2, 1)
    with pytest.raises(ValueError): training_loss(logits, target, **args)
    assert logits.grad is None

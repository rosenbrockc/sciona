import math

import pytest
import torch

from sciona.dfdc_loss import balanced_bce


@pytest.mark.parametrize('targets', [[.01,.99,.99,.99], [.01,.01,.01,.99], [.99,.99], [.01,.01], [.5,.99]])
def test_independent_value_and_gradient(targets):
    values = [-1.4 + i*.8 for i in range(len(targets))]
    logits = torch.tensor(values,dtype=torch.float64).reshape(-1,1).requires_grad_()
    labels = torch.tensor(targets,dtype=torch.float64).reshape(-1,1)
    loss = balanced_bce(logits, labels)
    expected = 0.
    gradients = []
    for group in (False,True):
        indices = [i for i,y in enumerate(targets) if (y>.5)==group]
        if indices:
            expected += sum(math.log1p(math.exp(values[i]))-targets[i]*values[i] for i in indices)/(2*len(indices))
    for x,y in zip(values,targets):
        size = sum((t>.5)==(y>.5) for t in targets)
        gradients.append((1/(1+math.exp(-x))-y)/(2*size))
    assert loss.item() == pytest.approx(expected,abs=1e-14)
    loss.backward()
    torch.testing.assert_close(logits.grad[:,0],torch.tensor(gradients,dtype=torch.float64),rtol=1e-14,atol=1e-14)


def test_missing_class_keeps_half_weight():
    logits = torch.zeros(3,1)
    assert balanced_bce(logits,torch.ones_like(logits)).item() == pytest.approx(math.log(2)/2)


def test_class_repetition_does_not_change_class_weight():
    x=torch.tensor([[-1.],[2.]])
    y=torch.tensor([[.01],[.99]])
    repeated_x=torch.cat([x[:1],x[1:].repeat(9,1)])
    repeated_y=torch.cat([y[:1],y[1:].repeat(9,1)])
    torch.testing.assert_close(balanced_bce(x,y),balanced_bce(repeated_x,repeated_y))


@pytest.mark.parametrize('x,y', [
    (torch.empty(0,1),torch.empty(0,1)),
    (torch.zeros(2),torch.zeros(2)),
    (torch.zeros(1,1),torch.tensor([[1.1]])),
    (torch.tensor([[float('nan')]]),torch.zeros(1,1)),
    (torch.zeros(1,1),torch.tensor([[float('inf')]])),
])
def test_invalid_loss_inputs_rejected(x,y):
    with pytest.raises(ValueError):balanced_bce(x,y)

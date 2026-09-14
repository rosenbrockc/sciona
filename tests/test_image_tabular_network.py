import pytest
import torch
from sciona.image_tabular_network import HybridHead,smoothed_loss


def test_hand_smoothed_loss_and_gradient():
    logits=torch.tensor([-2.,0.,3.],dtype=torch.float64,requires_grad=True)
    labels=torch.tensor([0.,1.,1.],dtype=torch.float64)
    targets=torch.tensor([.1,.9,.9],dtype=torch.float64)
    expected=(torch.nn.functional.softplus(logits)-targets*logits).mean()
    actual=smoothed_loss(logits,labels,smoothing=.2)
    torch.testing.assert_close(actual,expected)
    actual.backward()
    torch.testing.assert_close(logits.grad,(torch.sigmoid(logits)-targets)/3)


def test_dropout_training_and_evaluation():
    torch.manual_seed(3);model=HybridHead(4,hidden=12,dropout=.5).double()
    features=torch.ones(8,4,dtype=torch.float64)
    model.train();first=model(features);second=model(features)
    assert not torch.equal(first,second)
    model.eval();first=model(features);second=model(features)
    torch.testing.assert_close(first,second,rtol=0,atol=0)
    first.square().sum().backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())

@pytest.mark.parametrize('smoothing',[-.1,1.,True,float('nan')])
def test_invalid_smoothing(smoothing):
    with pytest.raises(ValueError):smoothed_loss(torch.zeros(2),torch.tensor([0.,1.]),smoothing=smoothing)

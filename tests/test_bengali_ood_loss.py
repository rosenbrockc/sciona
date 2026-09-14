import math
import pytest
import torch
from sciona.bengali_ood_loss import ood_training_loss


def test_positive_retained_and_normalization_is_per_sample():
    logits=torch.full((2,1295),-20.)
    logits[0,0]=-10;logits[0,1]=0
    logits[1,2]=-10;logits[1,3:5]=0
    logits.requires_grad_()
    loss=ood_training_loss(logits,torch.tensor([0,2]))
    positive=math.log1p(math.exp(10))
    assert loss.item()==pytest.approx(((positive+math.log(2))/2+(positive+2*math.log(2))/3)/2)
    loss.backward()
    assert logits.grad[0,0].item()==pytest.approx((1/(1+math.exp(10))-1)/4)
    assert logits.grad[0,1].item()==pytest.approx(.125)
    assert logits.grad[1,3].item()==pytest.approx(1/12)
    assert torch.count_nonzero(logits.grad)==5


def test_low_confidence_negatives_have_no_gradient_even_when_all_excluded():
    logits=torch.full((1,1295),-100.,requires_grad=True)
    loss=ood_training_loss(logits,torch.tensor([1294]));loss.backward()
    assert loss.item()==100.
    assert torch.count_nonzero(logits.grad)==1 and logits.grad[0,1294]==-1


@pytest.mark.parametrize('labels',[torch.tensor([-1]),torch.tensor([1295]),torch.tensor([1.]),torch.tensor([1,2])])
def test_invalid_targets_rejected(labels):
    with pytest.raises(ValueError):ood_training_loss(torch.zeros(1,1295),labels)

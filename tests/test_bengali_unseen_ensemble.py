import pytest
import torch
from sciona.bengali_unseen_ensemble import marginal_predictions


def test_marginal_decision_differs_from_joint_argmax():
    logits=torch.full((1,14784),-100.)
    logits[0,[0,96,89]]=torch.tensor([.4,.35,.25]).log()
    result=marginal_predictions(logits,logits)
    assert logits.argmax(1).item()==0
    assert result['joint_classes'].item()==88
    assert result['components'].tolist()==[[1,0,0]]


def test_consonant_remap_only_at_submission_boundary():
    logits=torch.full((1,14784),-100.);logits[0,14783]=0
    result=marginal_predictions(logits,logits)
    assert result['joint_classes'].item()==14783
    assert result['components'].tolist()==[[167,10,7]]
    assert result['submission_components'].tolist()==[[167,10,2]]


def test_each_branch_softmax_precedes_combination():
    a=torch.full((1,14784),-100.);a[0,0]=0;a[0,88]=1
    b=torch.full((1,14784),-100.);b[0,0]=2;b[0,88]=0
    expected=marginal_predictions(a,b)
    shifted=marginal_predictions(a+500,b-500)
    torch.testing.assert_close(expected['components'],shifted['components'])
    assert expected['components'][0,0]==0


def test_misaligned_or_nonfinite_logits_rejected():
    with pytest.raises(ValueError):marginal_predictions(torch.zeros(1,14784),torch.zeros(2,14784))
    with pytest.raises(ValueError):marginal_predictions(torch.full((1,14784),float('nan')),torch.zeros(1,14784))

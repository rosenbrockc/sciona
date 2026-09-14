import hashlib
import pytest
import torch
from torch import nn
from sciona.bengali_checkpoint_handoff import load_frozen_classifier


def save(path,state):
    torch.save(state,path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_verified_handoff_freezes_complete_state_and_preserves_input_gradients(tmp_path):
    source=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3))
    target=nn.Sequential(nn.Linear(3,3),nn.BatchNorm1d(3))
    digest=save(tmp_path/'weights.pt',source.state_dict())
    loaded=load_frozen_classifier(target,tmp_path/'weights.pt',expected_sha256=digest)
    assert loaded is target and all(not m.training for m in target.modules())
    assert all(not p.requires_grad for p in target.parameters())
    for key,value in source.state_dict().items():torch.testing.assert_close(value,target.state_dict()[key],rtol=0,atol=0)
    image=torch.ones(2,3,requires_grad=True);target(image).square().sum().backward()
    assert torch.isfinite(image.grad).all() and image.grad.abs().sum()>0


@pytest.mark.parametrize('corruption',['hash','missing','shape','dtype','nan'])
def test_rejected_checkpoint_does_not_mutate_model(tmp_path,corruption):
    model=nn.Linear(3,2);original={k:v.clone() for k,v in model.state_dict().items()}
    state={k:v.clone()+1 for k,v in original.items()}
    if corruption=='missing':del state['bias']
    elif corruption=='shape':state['bias']=torch.zeros(3)
    elif corruption=='dtype':state['bias']=state['bias'].double()
    elif corruption=='nan':state['bias'][0]=float('nan')
    digest=save(tmp_path/'weights.pt',state)
    if corruption=='hash':digest='0'*64
    with pytest.raises(ValueError):load_frozen_classifier(model,tmp_path/'weights.pt',expected_sha256=digest)
    for key,value in model.state_dict().items():torch.testing.assert_close(value,original[key],rtol=0,atol=0)
    assert model.training and all(p.requires_grad for p in model.parameters())

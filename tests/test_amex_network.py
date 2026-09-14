"""Synthetic last-timestep pooling and branch routing checks."""
import torch
import pytest
from sciona.amex_network import Network


def fixture():
    torch.manual_seed(112)
    x=torch.randn(3,13,4);mask=torch.arange(13)[None,:]<torch.tensor([3,13,1])[:,None];f=torch.randn(3,6)
    return x,mask.float(),f


def test_pool_matches_individual_unpadded_sequence_last_output():
    x,m,f=fixture();net=Network(4,6,hidden_width=8).eval();projected=net.sequence_projection(x)
    result=net.pool(projected,m)
    expected=[]
    for i,n in enumerate((3,13,1)):
        outputs,_=net.recurrence(projected[i:i+1,:n]);expected.append(outputs[0,-1])
    torch.testing.assert_close(result,torch.stack(expected),rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('combined',[False,True])
def test_masked_padding_invariance_and_gradients(combined):
    x,m,f=fixture();net=Network(4,6,combined=combined,hidden_width=8).eval()
    y=net(x,m,f);altered=x.clone();altered[m==0]=1000
    torch.testing.assert_close(y,net(altered,m,f),atol=1e-7,rtol=1e-6)
    assert y.shape==(3,1) and ((y>0)&(y<1)).all()
    y.sum().backward();assert net.recurrence.weight_ih_l0.grad is not None
    assert (net.feature_projection[0].weight.grad is not None)==combined


def test_invalid_masks_and_shapes():
    x,m,f=fixture();net=Network(4,6)
    m[0,0]=0
    with pytest.raises(ValueError):net(x,m,f)
    with pytest.raises(ValueError):net(x[:,:,:3],m,f)

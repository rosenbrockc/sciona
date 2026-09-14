import pytest
import torch
from sciona.biosignal_sequence_network import DualSignalHead,mask_spectrogram


def test_both_branches_receive_gradients_and_batch_independence():
    torch.manual_seed(4);net=DualSignalHead(2,5,width=3).double()
    wave=torch.randn(3,2,24,dtype=torch.float64,requires_grad=True)
    spectral=torch.randn(3,2,5,7,dtype=torch.float64,requires_grad=True)
    logits=net(wave,spectral)
    assert logits.shape==(3,)
    torch.testing.assert_close(net(wave[:1],spectral[:1]),logits[:1])
    logits.square().sum().backward()
    assert wave.grad.abs().sum()>0 and spectral.grad.abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())


def test_mask_bounds_repeat_and_input_unchanged():
    x=torch.ones(12,2,8,9,dtype=torch.float64)
    first=mask_spectrogram(x,max_frequency=3,max_time=4,generator=torch.Generator().manual_seed(5))
    second=mask_spectrogram(x,max_frequency=3,max_time=4,generator=torch.Generator().manual_seed(5))
    assert torch.equal(first,second) and torch.equal(x,torch.ones_like(x))
    assert (first==0).any() and (first==1).any()
    for row in first:
        assert int((row[0]==0).all(1).sum())<=3
        assert int((row[0]==0).all(0).sum())<=4
        assert torch.equal(row[0],row[1])


def test_zero_masks_identity():
    x=torch.randn(2,1,5,7)
    torch.testing.assert_close(mask_spectrogram(x,max_frequency=0,max_time=0,generator=torch.Generator()),x)

@pytest.mark.parametrize('frequency,time',[(5,0),(0,7),(-1,1),(True,0)])
def test_invalid_masks(frequency,time):
    with pytest.raises(ValueError):mask_spectrogram(torch.ones(1,1,5,7),max_frequency=frequency,max_time=time,generator=torch.Generator())

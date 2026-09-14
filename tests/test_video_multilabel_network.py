import numpy as np
import pytest
import torch
from sciona.video_multilabel_network import VideoHead,pad_sequences


def test_masked_uniform_attention_and_context_formula():
    net=VideoHead(2,3,2).double()
    with torch.no_grad():
        for p in net.attention.parameters():p.zero_()
        net.context.weight.zero_();net.context.bias.zero_()
        net.classifier.weight.copy_(torch.eye(2));net.classifier.bias.zero_()
    frames,mask=pad_sequences([[[1.,2.],[3.,6.]],[[9.,4.]]])
    pooled,weights=net.pool(frames,mask)
    torch.testing.assert_close(pooled,torch.tensor([[2.,4.],[9.,4.]],dtype=torch.float64))
    torch.testing.assert_close(weights,torch.tensor([[.5,.5],[1.,0.]],dtype=torch.float64))
    torch.testing.assert_close(net(frames,mask),pooled*.5)


def test_nonuniform_attention_matches_scalar_calculation():
    net=VideoHead(1,1,1).double()
    with torch.no_grad():
        net.attention[0].weight.fill_(2);net.attention[0].bias.fill_(.1)
        net.attention[2].weight.fill_(3);net.attention[2].bias.fill_(-.2)
    frames,mask=pad_sequences([[[.2],[.7]]])
    pooled,weights=net.pool(frames,mask)
    scores=3*np.tanh(2*np.array([.2,.7])+.1)-.2
    expected=np.exp(scores-scores.max());expected/=expected.sum()
    np.testing.assert_allclose(weights.detach().numpy()[0],expected)
    assert pooled.item()==pytest.approx(expected@np.array([.2,.7]))


def test_padding_batch_and_permutation_invariance():
    torch.manual_seed(9);net=VideoHead(2,4,3).double()
    frames,mask=pad_sequences([[[1.,2.],[3.,4.]],[[5.,6.]]])
    expected=net(frames,mask)
    frames[1,1]=float('nan')
    torch.testing.assert_close(net(frames,mask),expected)
    torch.testing.assert_close(net(frames.flip(1),mask.flip(1)),expected)
    torch.testing.assert_close(net(frames[:1],mask[:1]),expected[:1])


def test_gradients_ignore_padding():
    torch.manual_seed(3);net=VideoHead(2,3,2).double()
    frames,mask=pad_sequences([[[1.,2.],[3.,1.]],[[2.,5.]]]);frames.requires_grad_()
    net(frames,mask).square().sum().backward()
    assert torch.equal(frames.grad[~mask],torch.zeros_like(frames.grad[~mask]))
    assert frames.grad[mask].abs().sum()>0
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in net.parameters())

@pytest.mark.parametrize('sequences',[[],[[]],[[[]]],[[[True]]],[[[float('nan')]]],[[[1.],[1.,2.]]]])
def test_bad_sequences(sequences):
    with pytest.raises(ValueError):pad_sequences(sequences)


def test_empty_mask_rejected():
    with pytest.raises(ValueError):VideoHead(2,3,2).double()(torch.zeros(1,2,2,dtype=torch.float64),torch.zeros(1,2,dtype=torch.bool))

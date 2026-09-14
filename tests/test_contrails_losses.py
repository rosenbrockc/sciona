"""Independent objective and gradient checks on synthetic segmentation tensors."""
import numpy as np
import pytest
import torch

from sciona.contrails_losses import BCELoss, DiceLoss


@pytest.mark.parametrize('weights', [[0., 0.], [1., 1.], [0., 1.], [.2, .7]])
@pytest.mark.parametrize('scale', [1., 100.])
def test_bce_value_and_gradient(weights, scale):
    rng = np.random.default_rng(1205)
    a = torch.tensor(rng.normal(size=(2, 1, 3, 3)) * scale, requires_grad=True)
    b = torch.tensor(rng.normal(size=(2, 1, 2, 2)) * scale, requires_grad=True)
    ya = torch.tensor(rng.uniform(size=a.shape))
    yb = torch.tensor(rng.uniform(size=b.shape))
    w = torch.tensor(weights, dtype=torch.float64)
    result = BCELoss()(a, ya, b, yb, w)
    aa, bb = a.detach().numpy(), b.detach().numpy()
    # Stable logistic objective, independently evaluated with NumPy.
    loss_a = np.logaddexp(0, aa) - aa * ya.numpy()
    loss_b = np.logaddexp(0, bb) - bb * yb.numpy()
    expected = np.mean(loss_a.mean((1, 2, 3)) + np.array(weights) * loss_b.mean((1, 2, 3)))
    np.testing.assert_allclose(result.item(), expected, rtol=1e-12)
    result.backward()
    expected_a = (1 / (1 + np.exp(-aa)) - ya.numpy()) / aa.size
    expected_b = (1 / (1 + np.exp(-bb)) - yb.numpy()) * np.array(weights)[:, None, None, None] / bb.size
    np.testing.assert_allclose(a.grad.numpy(), expected_a, atol=1e-15)
    np.testing.assert_allclose(b.grad.numpy(), expected_b, atol=1e-15)


def test_dice_uses_global_weighted_mass_not_average_per_image():
    a = torch.tensor([[[[-2., 1.]]], [[[3., 0.]]]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[[[-1., 2.]]], [[[0., 4.]]]], dtype=torch.float64, requires_grad=True)
    ya = torch.tensor([[[[0., 1.]]], [[[1., 0.]]]], dtype=torch.float64)
    yb = 1 - ya
    w = torch.tensor([0., .3], dtype=torch.float64)
    actual = DiceLoss()(a, ya, b, yb, w)
    pa, pb = a.detach().sigmoid().numpy(), b.detach().sigmoid().numpy()
    numerator = 2 * (np.sum(pa * ya.numpy()) + .3 * np.sum(pb[1] * yb[1].numpy()))
    denominator = np.sum(pa + ya.numpy()) + .3 * np.sum(pb[1] + yb[1].numpy())
    np.testing.assert_allclose(actual.item(), 1 - numerator / denominator, rtol=1e-12)
    actual.backward()
    assert torch.count_nonzero(b.grad[0]) == 0
    assert torch.count_nonzero(b.grad[1]) > 0
    assert torch.autograd.gradcheck(lambda x, y: DiceLoss()(x, ya, y, yb, w), (a, b))


def test_empty_positive_dice_is_one_and_finite():
    x = torch.full((2, 1, 2, 2), -1000.)
    target = torch.zeros_like(x)
    assert DiceLoss()(x, target, x, target, 0.).item() == 1.

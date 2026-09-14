"""Synthetic numerical oracles; no competition data or pretrained weights."""
import numpy as np
import pytest
import torch
from scipy.optimize import brentq

from sciona.cassava_loss import vit_loss


def oracle(row, target):
    # Solve sum(exp_t(logit-normalizer))=1 independently of fixed-point code.
    z = np.asarray(row) - max(row)
    root = brentq(lambda n: sum((1 - .4 * (z - n)) ** -2.5) - 1, 0, 100)
    p = (1 - .4 * (z - root)) ** -2.5
    y = np.asarray(target) * .925 + .015
    return sum(y * ((y + 1e-10) ** .2 - p ** .2) / .2 - (y ** 1.2 - p ** 1.2) / 1.2)


def test_converged_objective_and_gradient_against_root_solver():
    rng = np.random.default_rng(73)
    rows = rng.normal(size=(12, 5)) * 3
    targets = rng.dirichlet(np.ones(5), size=12)
    x = torch.tensor(rows, requires_grad=True)
    result = vit_loss(x, torch.tensor(targets), iterations=100)
    np.testing.assert_allclose(result.detach(), [oracle(r, y) for r, y in zip(rows, targets)], atol=1e-11)
    weights = torch.linspace(.2, 1.7, 12, dtype=torch.float64)
    (result * weights).sum().backward()
    expected = np.zeros_like(rows)
    for i in range(12):
        for j in range(5):
            plus, minus = rows[i].copy(), rows[i].copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            expected[i, j] = (oracle(plus, targets[i]) - oracle(minus, targets[i])) / 2e-5 * weights[i].item()
    np.testing.assert_allclose(x.grad, expected, atol=2e-8, rtol=1e-6)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_shift_invariance_and_trainable_default(dtype):
    x = torch.tensor([[1., -2., 3., 0., 2.]], dtype=dtype, requires_grad=True)
    y = torch.tensor([[0., 1., 0., 0., 0.]], dtype=dtype)
    before = vit_loss(x, y)
    torch.testing.assert_close(before, vit_loss(x + 100, y))
    before.sum().backward()
    assert torch.isfinite(x.grad).all()
    assert abs(x.grad.sum().item()) < 1e-6
    assert vit_loss(x.detach() - .1 * x.grad, y).item() < before.item()


def test_uniform_logits_have_analytical_smoothed_loss():
    x = torch.zeros((1, 5), dtype=torch.float64, requires_grad=True)
    y = torch.tensor([[1., 0., 0., 0., 0.]], dtype=torch.float64)
    assert vit_loss(x, y).item() == pytest.approx(oracle([0.] * 5, y[0].numpy()), abs=1e-12)


@pytest.mark.parametrize('kind', ['shape', 'nan', 'negative', 'unnormalized', 'target_grad', 'half', 'iterations', 'overflow'])
def test_rejects_invalid_contract(kind):
    x = torch.zeros((2, 5), dtype=torch.float32)
    y = torch.full_like(x, .2)
    iterations = 5
    if kind == 'shape': x = x[:, :4]
    if kind == 'nan': x[0, 0] = float('nan')
    if kind == 'negative': y[0, 0] = -.2
    if kind == 'unnormalized': y *= .5
    if kind == 'target_grad': y.requires_grad_()
    if kind == 'half': x, y = x.half(), y.half()
    if kind == 'iterations': iterations = True
    if kind == 'overflow': x[0, :2] = torch.tensor([3e38, -3e38])
    with pytest.raises(ValueError): vit_loss(x, y, iterations=iterations)

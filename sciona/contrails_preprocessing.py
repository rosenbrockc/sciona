"""In-memory preprocessing for Jun Koda's Contrails solution.

Adapted from public source commit 08a15beb36f9cbed4c3990e74c625c1332b61fe8.
Copyright (c) 2023 Jun Koda; MIT, see docs/licenses/Contrails-MIT.txt.
These components are part of the pending full execution graph, not a published CDG.
"""
import torch


def ash_color(thermal_triplet: torch.Tensor) -> torch.Tensor:
    """Transform (..., 3, H, W) selected thermal inputs to inverted false color.

    Preserves the source's unbounded output: no clipping or per-image scaling.
    Inputs must already be in the source's physical temperature scale and order.
    """
    x = thermal_triplet
    if not isinstance(x, torch.Tensor) or x.ndim < 3 or x.shape[-3] != 3:
        raise ValueError('Expected tensor (..., 3, H, W)')
    if not x.is_floating_point() or not torch.isfinite(x).all():
        raise ValueError('Expected finite floating point temperatures')
    r = (x[..., 2, :, :] - x[..., 1, :, :] + 4) / 6
    g = (x[..., 1, :, :] - x[..., 0, :, :] + 4) / 9
    b = (x[..., 1, :, :] - 243) / 60
    return 1 - torch.stack((r, g, b), dim=-3)


class SpatialTTA:
    """Source D4/rotation TTA, preserving probability versus logit averaging.

    Square BCHW inputs; inverse averaging expects a single output channel.
    Source temporal model preparation is a separate operation.
    """
    _modes = {'d4prob': (8, True), 'd4logit': (8, False),
              'rotprob': (4, True), 'rotlogit': (4, False), 'none': (1, False)}

    def __init__(self, mode: str):
        if mode not in self._modes:
            raise ValueError('Unknown spatial TTA mode')
        self.n, self.prob = self._modes[mode]

    @staticmethod
    def _check(x):
        if not isinstance(x, torch.Tensor) or x.ndim != 4 or min(x.shape) < 1:
            raise ValueError('Expected nonempty BCHW tensor')
        if x.shape[2] != x.shape[3] or not x.is_floating_point() or not torch.isfinite(x).all():
            raise ValueError('Expected finite floating point square images')

    def stack(self, x):
        self._check(x)
        if self.n == 1:
            return x
        stack = []
        for k in range(4):
            xa = torch.rot90(x, k, dims=[2, 3])
            stack.append(xa)
            if self.n == 8:
                stack.append(torch.flip(xa, dims=[3]))
        return torch.cat(stack, dim=0)

    def average(self, y):
        self._check(y)
        if y.shape[1] != 1 or y.shape[0] % self.n:
            raise ValueError('Expected one channel and complete augmentation groups')
        if self.n == 1:
            return y
        batch, channels, height, width = y.shape
        pred = y.reshape(self.n, batch // self.n, channels, height, width)
        # Source accumulates in float32 even when predictions have another dtype.
        avg = torch.zeros((batch // self.n, 1, height, width), dtype=torch.float32, device=y.device)
        if self.prob:
            pred = pred.sigmoid()
        for k in range(4):
            if self.n == 4:
                avg += (1 / self.n) * torch.rot90(pred[k], -k, dims=[2, 3])
            else:
                avg += (1 / self.n) * torch.rot90(pred[2 * k], -k, dims=[2, 3])
                avg += (1 / self.n) * torch.rot90(pred[2 * k + 1].flip(dims=[3]), -k, dims=[2, 3])
        return avg.clamp(1e-6, 1 - 1e-6).logit() if self.prob else avg

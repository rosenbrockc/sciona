"""Differentiable source Web Traffic losses expressed in PyTorch.

MIT model.py adaptation; see docs/licenses/WebTraffic-MIT.txt.
Weighted reduction follows TensorFlow 1.10 SUM_BY_NONZERO_WEIGHTS semantics.
"""
import torch


def weighted_loss(losses, weights):
    weighted = losses.float() * weights.float()
    present = torch.count_nonzero(torch.broadcast_to(weights, losses.shape))
    result = weighted.sum() / present.clamp(min=1)
    return torch.where(present > 0, result, torch.zeros_like(result)).to(losses.dtype)


def calc_loss(predictions, true_y, additional_mask=None):
    mask = torch.isfinite(true_y)
    true_y = torch.where(mask, true_y, torch.zeros_like(true_y))
    weights = mask.float()
    if additional_mask is not None:
        weights = weights * additional_mask.unsqueeze(0)
    mae = weighted_loss(torch.abs(true_y.float() - predictions.float()), weights)
    true_o, pred_o = torch.expm1(true_y), torch.expm1(predictions)
    denominator = (true_o.abs() + pred_o.abs() + .1).clamp(min=.6)
    smooth = weighted_loss((pred_o - true_o).abs() / denominator * 2., weights)
    true_round = torch.round(true_o)
    pred_round = torch.round(pred_o).clamp(min=0.)
    total = true_round.abs() + pred_round.abs()
    raw = (pred_round - true_round).abs() / total * 2.
    rounded = torch.where(total < .01, torch.zeros_like(total), raw)
    score = (rounded * weights).sum() / weights.sum()
    return mae, smooth, score, true_y.numel()


def decode_predictions(readout, mean, std):
    return readout.transpose(0, 1) * std.unsqueeze(-1) + mean.unsqueeze(-1)


def rnn_stability_loss(output, beta):
    if beta == 0.:
        return 0.
    lengths = torch.sqrt(torch.sum(output.square(), dim=-1))
    return beta * torch.mean((lengths[1:] - lengths[:-1]).square())


def rnn_activation_loss(output, beta):
    if beta == 0.:
        return 0.
    return output.square().sum() * .5 * beta

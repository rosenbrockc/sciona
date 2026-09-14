"""Source-supported least-squares and cycle objectives for Bengali CycleGAN.

Inputs are discriminator outputs and reconstructed images. The caller owns
replay, frozen-classifier guidance, gradient isolation and optimizer ordering.
This module alone is not a training implementation.
"""
import torch
from torch.nn import functional as F


def _finite(value, name):
    if (not isinstance(value, torch.Tensor) or value.dtype != torch.float32
            or value.device.type != 'cpu' or not value.numel()
            or not torch.isfinite(value).all()):
        raise ValueError(f'{name} must be a nonempty finite float32 CPU tensor')


def least_squares(prediction, *, real):
    _finite(prediction, 'prediction')
    if type(real) is not bool:
        raise ValueError('real must be boolean')
    return F.mse_loss(prediction, torch.full_like(prediction, float(real)))


def generator_objective(*, fake_a_prediction, fake_b_prediction,
                        reconstructed_a, real_a, reconstructed_b, real_b,
                        weighted_font_loss):
    for name, value in [('weighted_font_loss', weighted_font_loss),
                        ('reconstructed_a', reconstructed_a), ('real_a', real_a),
                        ('reconstructed_b', reconstructed_b), ('real_b', real_b)]:
        _finite(value, name)
    if weighted_font_loss.ndim != 0 or weighted_font_loss.item() < 0:
        raise ValueError('weighted font loss must be a nonnegative scalar')
    for reconstruction, original in [(reconstructed_a, real_a), (reconstructed_b, real_b)]:
        if reconstruction.shape != original.shape or original.ndim != 4 or original.shape[1] != 3:
            raise ValueError('cycle images must have matching NCHW RGB shapes')
    terms = dict(gan_a=least_squares(fake_a_prediction, real=True),
                 gan_b=least_squares(fake_b_prediction, real=True),
                 cycle_a=10 * F.l1_loss(reconstructed_a, real_a),
                 cycle_b=10 * F.l1_loss(reconstructed_b, real_b),
                 font=weighted_font_loss)
    terms['total'] = terms['gan_a'] + terms['gan_b'] + terms['cycle_a'] + terms['cycle_b'] + terms['font']
    return terms


def discriminator_objective(*, real_a_prediction, replay_a_prediction,
                            real_b_prediction, replay_b_prediction):
    terms = dict(real_a=least_squares(real_a_prediction, real=True),
                 fake_a=least_squares(replay_a_prediction, real=False),
                 real_b=least_squares(real_b_prediction, real=True),
                 fake_b=least_squares(replay_b_prediction, real=False))
    terms['domain_a'] = (terms['real_a'] + terms['fake_a']) / 2
    terms['domain_b'] = (terms['real_b'] + terms['fake_b']) / 2
    terms['total'] = terms['domain_a'] + terms['domain_b']
    return terms

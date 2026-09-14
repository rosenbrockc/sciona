"""Explicit uncertainty masking and perturbation for OpenVaccine reconstruction.

The winner describes proportional random perturbation but does not specify its
probability law. Callers provide dimensionless draws and eligibility, avoiding
an invented historical distribution or sequence-position rule.
"""
import numpy as np


def uncertainty_targets(mean, uncertainty, eligible, *, maximum_uncertainty, perturbations):
    mean, uncertainty = np.asarray(mean), np.asarray(uncertainty)
    eligible, perturbations = np.asarray(eligible), np.asarray(perturbations)
    if mean.ndim != 3 or any(n < 1 for n in mean.shape) or mean.shape[-1] != 5:
        raise ValueError('Expected nonempty batch/position/five-target tensors')
    if any(x.shape != mean.shape for x in [uncertainty, eligible, perturbations]):
        raise ValueError('Uncertainty, eligibility and perturbations must match targets')
    if eligible.dtype != np.bool_:
        raise ValueError('Eligibility must be boolean')
    if any(x.dtype != np.float32 for x in [mean, uncertainty, perturbations]):
        raise ValueError('Numeric target tensors must be float32')
    if any(not np.isfinite(x).all() for x in [mean, uncertainty, perturbations]) or (uncertainty < 0).any():
        raise ValueError('Finite targets/draws and nonnegative uncertainties required')
    if isinstance(maximum_uncertainty, bool) or not np.isscalar(maximum_uncertainty):
        raise ValueError('Maximum uncertainty must be a finite nonnegative scalar')
    if not np.isfinite(maximum_uncertainty) or maximum_uncertainty < 0:
        raise ValueError('Maximum uncertainty must be a finite nonnegative scalar')
    retained = eligible & (uncertainty <= maximum_uncertainty)
    if not retained.any():
        raise ValueError('No eligible pseudo-labels remain')
    result = np.full(mean.shape, np.nan, dtype=np.float32)
    # Calculate only retained targets; excluded values need not be perturbed.
    values = mean[retained].astype(np.float64) + uncertainty[retained].astype(np.float64)*perturbations[retained].astype(np.float64)
    if not np.isfinite(values).all() or (abs(values) > np.finfo(np.float32).max).any():
        raise ValueError('Perturbed targets exceed finite float32 range')
    result[retained] = values
    return result


def validation_rollback_required(before, after, *, absolute_tolerance):
    values = (before, after, absolute_tolerance)
    if any(isinstance(x, bool) or not np.isscalar(x) or not np.isfinite(x) or x < 0 for x in values):
        raise ValueError('Finite nonnegative scores and absolute tolerance required')
    # Equality is accepted: the winner describes deterioration by more than
    # the threshold. The caller must explicitly define metric units.
    return bool(after > before + absolute_tolerance)

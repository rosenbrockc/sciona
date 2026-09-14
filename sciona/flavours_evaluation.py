"""Validated reuse of existing project metrics for Flavours reference execution.

Evaluation inputs are supplied separately from fitting. OOF discrimination is
diagnostic because upstream mass cross-fitting is not fully nested.
"""
import numpy as np
from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
    compute_cvm_mass_decorrelation, compute_ks_agreement, roc_auc_truncated_weighted,
)


def _vector(value, name, *, probability=False):
    a = np.asarray(value, dtype=np.float64)
    if a.ndim != 1 or a.size == 0 or not np.isfinite(a).all():
        raise ValueError(name + ' must be a nonempty finite vector')
    if probability and np.any((a < 0) | (a > 1)):
        raise ValueError(name + ' must lie in [0, 1]')
    return a


def evaluate(labels, oof_scores, quality, agreement_a, agreement_b,
             weights_a, weights_b, correlation_scores, correlation_mass):
    y = _vector(labels, 'labels')
    oof = _vector(oof_scores, 'OOF scores', probability=True)
    quality = _vector(quality, 'quality')
    a = _vector(agreement_a, 'agreement A', probability=True)
    b = _vector(agreement_b, 'agreement B', probability=True)
    wa = _vector(weights_a, 'weights A')
    wb = _vector(weights_b, 'weights B')
    corr = _vector(correlation_scores, 'correlation scores', probability=True)
    mass = _vector(correlation_mass, 'correlation mass')
    if y.shape != oof.shape or quality.shape != y.shape or not set(y).issubset({0, 1}):
        raise ValueError('Binary labels, quality and OOF predictions must align')
    selected = quality > .4
    if set(y[selected]) != {0, 1}:
        raise ValueError('Both classes required after strict source quality selection')
    for scores, weights in ((a, wa), (b, wb)):
        if scores.shape != weights.shape or np.any(weights < 0) or weights.sum() <= 0 or not np.isfinite(weights.sum()):
            raise ValueError('Aligned nonnegative weights with finite positive sum required')
    if corr.shape != mass.shape or len(corr) < 200:
        raise ValueError('Correlation evaluation requires at least 200 aligned rows')
    ks = compute_ks_agreement(a, b, wa, wb)
    cvm = compute_cvm_mass_decorrelation(corr, mass, n_neighbours=200, step=50)
    return dict(ks_agreement=float(ks), cvm_mass=float(cvm),
                truncated_weighted_auc=float(roc_auc_truncated_weighted(y[selected], oof[selected])),
                agreement_passed=bool(ks < .09), correlation_passed=bool(cvm < .002),
                oof_scope='Diagnostic only; upstream mass features are not nested within classifier folds')

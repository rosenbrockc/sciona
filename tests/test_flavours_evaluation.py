import numpy as np
import pytest
from sklearn.metrics import roc_curve
from sciona.atoms.ml.constrained_ml.decorrelation.atoms import (
    compute_cvm_mass_decorrelation, compute_ks_agreement, roc_auc_truncated_weighted,
)
from sciona.flavours_evaluation import evaluate


def test_reused_metrics_against_independent_cdf_and_roc_oracles():
    rng = np.random.default_rng(415)
    for ties in (False, True):
        a, b = rng.random(31), rng.random(29)
        if ties: a, b = np.round(a, 1), np.round(b, 1)
        wa, wb = rng.random(31), rng.random(29)
        thresholds = np.unique(np.r_[a,b])
        expected = max(abs(wa[a <= t].sum()/wa.sum() - wb[b <= t].sum()/wb.sum()) for t in thresholds)
        np.testing.assert_allclose(compute_ks_agreement(a,b,wa,wb), expected, atol=1e-14)
        y, scores = np.r_[np.zeros(31),np.ones(29)], np.r_[a,b]
        fpr,tpr,_ = roc_curve(y,scores)
        bounds = [0,.2,.4,.6,.8,1]
        area = sum(w*np.trapezoid(np.clip(tpr-lo,0,hi-lo), fpr)
                   for lo,hi,w in zip(bounds[:-1],bounds[1:],[4,3,2,1,0])) / 2
        np.testing.assert_allclose(roc_auc_truncated_weighted(y,scores),area,atol=1e-14)
    scores, mass = rng.random(350), rng.random(350)
    ranks = np.argsort(np.argsort(scores[np.argsort(mass)],kind='stable'),kind='stable')
    contributions = []
    for start in range(0,151,50):
        subset = ranks[start:start+200]
        contributions.append(np.mean([(np.mean(subset <= k)-(k+1)/350)**2 for k in range(350)]))
    np.testing.assert_allclose(compute_cvm_mass_decorrelation(scores,mass),np.mean(contributions),atol=1e-14)


def test_source_quality_cut_is_strict_and_small_correlation_input_rejected():
    args = [[0,1,0,1],[0,1,1,0],[1,1,.4,.4], [0,1],[0,1],[1,1],[1,1],
            np.linspace(0,1,200),np.arange(200)]
    assert evaluate(*args)['truncated_weighted_auc'] == 1
    args[-1],args[-2] = np.arange(199),np.linspace(0,1,199)
    with pytest.raises(ValueError): evaluate(*args)

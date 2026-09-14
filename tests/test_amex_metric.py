"""Synthetic scalar weighted-ranking oracles."""
import numpy as np
import pytest
from sciona.amex_metric import score


def oracle(y,p):
    def area(order):
        total=sum(20 if z==0 else 1 for z in y);positives=sum(y);cw=cp=0.;value=0.
        for i in order:
            w=20 if y[i]==0 else 1;cw+=w/total;cp+=y[i]*w
            value+=(cp/positives-cw)*w
        return value
    order=np.argsort(p)[::-1];cut=int(.04*sum(20 if z==0 else 1 for z in y));weight=positive=0
    for i in order:
        weight+=20 if y[i]==0 else 1
        if weight<=cut:positive+=y[i]
    return .5*(area(order)/area(np.argsort(y)[::-1])+positive/sum(y))


def test_scalar_rank_oracle_and_ties():
    rng=np.random.default_rng(114)
    for _ in range(100):
        y=rng.integers(0,2,100);p=rng.integers(0,8,100)/7
        assert score(y,p)==pytest.approx(oracle(y,p),abs=1e-14)


def test_invalid_and_perfect():
    y=[0]*100+[1]*10
    assert score(y,y)==pytest.approx(1.)
    with pytest.raises(ValueError):score([1,1],[.2,.3])

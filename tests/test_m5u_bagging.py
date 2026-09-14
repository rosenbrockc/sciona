import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch
from sciona.m5u_bagging import train


def bag(seed):
    rng=np.random.default_rng(seed);x=rng.normal(size=(150,2))
    return dict(features=pd.DataFrame(x,columns=['a','b']),targets=2+x[:,0]*.3,
                groups=np.repeat([1,2,3],50))


def test_native_bags_quantiles_and_nested_holdout_isolation():
    data=[bag(3),bag(4)];models=train(data,[.1,.5,.9],[1.,1.,1.],12,seed=61)
    assert len(models)==18
    for item in models:
        groups=data[item.bag]['groups']
        assert (groups[item.training_rows]!=item.group).all()
        assert (groups[item.holdout_rows]==item.group).all()
        assert item.report['candidate_fits']==8 and item.report['refits']==1
        assert np.isfinite(item.report['outer_holdout_loss'])
    assert {(m.bag,m.group,m.quantile) for m in models}=={(b,g,q) for b in (0,1) for g in (1,2,3) for q in (.1,.5,.9)}


def test_insufficient_nested_groups_reject_before_search():
    data=bag(1);data['groups'][:]=np.tile([1,2],75)
    with patch('sciona.m5u_bagging.search') as search:
        with pytest.raises(ValueError,match='three groups'):train([data],[.5],[1.],12)
        search.assert_not_called()

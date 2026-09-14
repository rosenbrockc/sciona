import numpy as np
import pytest
from sciona.porto_tree_blend import fit_tree,blend


def test_real_raw_tree_learning():
    rng=np.random.default_rng(4);y=np.tile([0,1],100);x=y[:,None]*10+rng.normal(0,.2,(200,4))
    q=np.array([x[y==i].mean(axis=0) for i in (0,1)])
    c=dict(rounds=12,num_leaves=3,learning_rate=.2,min_data_in_leaf=2,feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,lambda_l2=0.)
    r=fit_tree(x,y,q,seed=7,controls=c)
    assert r['boosting_iterations']==12 and r['probabilities'][0]<.1 and r['probabilities'][1]>.9


def test_exact_arithmetic_mean_not_rank_blend():
    neural=np.array([[.1,.9],[.2,.8],[.3,.7],[.4,.6],[.5,.5]])
    np.testing.assert_allclose(blend(neural,np.array([.6,.4])),[.35,.65],rtol=0,atol=1e-15)


@pytest.mark.parametrize('problem',['missing_model','misaligned_tree','not_probability','nan'])
def test_invalid_ensemble(problem):
    a=np.full((5,3),.5);b=np.full(3,.5)
    if problem=='missing_model':a=a[:4]
    elif problem=='misaligned_tree':b=b[:2]
    elif problem=='not_probability':a[0,0]=2
    else:a[0,0]=np.nan
    with pytest.raises(ValueError):blend(a,b)

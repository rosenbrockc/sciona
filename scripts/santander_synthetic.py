"""Synthetic-only complete Santander graph fixture with full pseudo counts."""
import numpy as np


def payload():
    rng=np.random.default_rng(81);y=np.tile([0,1],30)
    x=y[:,None]*2+rng.normal(0,.5,(60,4))
    q=np.linspace(-1,3,8500)[:,None]*np.array([1.,1.1,.9,.8])[None,:]
    neural=dict(folds=10,seeds=[42],fold_seed=42,epochs=15,batch_size=1024,maximum_lr=.01)
    tree=dict(folds=10,seeds=[42],fold_seed=42,controls=dict(num_leaves=3,learning_rate=.1,feature_fraction=1.,
        bagging_fraction=1.,bagging_freq=0,min_data_in_leaf=2,max_rounds=8,stopping_rounds=2,categorical=True))
    return dict(version=1,training=dict(values=x.tolist(),labels=y.tolist(),identities=[f'r{i}' for i in range(len(x))]),
        query=dict(values=q.tolist(),identities=[f'q{i}' for i in range(len(q))]),
        controls=dict(initial_neural=neural,neural=dict(neural),tree=tree,reference_policy='retain_selected'))

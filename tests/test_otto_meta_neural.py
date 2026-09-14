import numpy as np
from sciona.otto_meta_neural import fit_bag
from sciona.otto_meta_xgboost import fit_bag as fit_tree
from sciona.otto_meta_adaboost import fit_bag as fit_ada
from sciona.otto_final_blend import blend


def test_all_600_native_meta_networks_and_real_three_family_blend():
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),20)
    x=np.eye(9)[y]*10+rng.uniform(0,.2,size=(180,9))-2;q=np.eye(9)*10-1.9
    neural=fit_bag(x,y,q,seed=12,controls=dict(hidden=[16],epochs=20,batch_size=64,learning_rate=.1,momentum=.9,ddof=1))
    assert neural.shape==(600,9,9)
    np.testing.assert_allclose(neural.sum(axis=2),1.,rtol=0,atol=1e-12)
    np.testing.assert_array_equal(neural.mean(axis=0).argmax(axis=1),np.arange(9))
    tree=fit_tree(x,y,q,seed=12,controls=dict(rounds=8,max_depth=3,eta=.4,subsample=.8,colsample_bytree=.8))
    ada=fit_ada(x,y,q,seed=12,controls=dict(algorithm='SAMME',boost_rounds=4,learning_rate=.7,trees=3,max_depth=3,min_samples_leaf=2,max_features=9))
    result=blend(tree,neural,ada)
    expected=.85*tree.mean(axis=0)**.65*neural.mean(axis=0)**.35+.15*ada.mean(axis=0)
    np.testing.assert_allclose(result['raw_scores'],expected)
    np.testing.assert_allclose(np.sum(result['probabilities'],axis=1),1.)
    np.testing.assert_array_equal(result['classes'],np.arange(9))

import numpy as np
from sciona.santander_lifecycle import fit
from sciona.santander_pseudo_labels import select_pseudo_labels
from sciona.santander_blend import blend_neural_tree


def test_complete_real_initial_training_pseudo_retraining_and_blend():
    rng=np.random.default_rng(81);y=np.tile([0,1],30)
    x=y[:,None]*2+rng.normal(0,.5,(60,4));q=rng.normal(1,1,(48,4))
    neural=dict(folds=10,seeds=(42,),epochs=15,batch_size=1024)
    tree=dict(folds=10,seeds=(42,),controls=dict(num_leaves=3,learning_rate=.1,feature_fraction=1.,
        bagging_fraction=1.,bagging_freq=0,min_data_in_leaf=2,max_rounds=8,stopping_rounds=2,categorical=True))
    result=fit(x,y,q,initial_neural_controls=neural,neural_controls=neural,tree_controls=tree,
        reference_policy='retain_selected',neural_positive=3,neural_negative=2,tree_positive=2,tree_negative=1)
    assert len(result['initial_neural']['models'])==10
    for family in ('neural','tree'):assert len(result['branches'][family]['models'])==10
    for family,positive,negative in [('neural',3,2),('tree',2,1)]:
        positions,labels=select_pseudo_labels(result['initial_neural']['mean_ranks'],positives=positive,negatives=negative)
        assert len(positions)==positive+negative
        assert result['selection_counts'][family]['labeled_rows']==60+len(positions)
    expected=blend_neural_tree(result['branches']['neural']['mean_ranks'],result['branches']['tree']['mean_ranks'])
    np.testing.assert_array_equal(result['scores'],expected)
    assert result['scores'].shape==(48,) and np.isfinite(result['scores']).all()

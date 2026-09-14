import numpy as np
import pytest
from sciona import santander_retraining as module


def sample():
    rng=np.random.default_rng(31)
    return rng.integers(0,9,(40,4)).astype(float),np.arange(40)%2,rng.integers(0,12,(12,4)).astype(float),np.arange(12,dtype=float)


def test_selected_rows_labels_and_unchanged_query_reference():
    x,y,q,s=sample();result=module.expanded_population(x,y,q,s,positives=3,negatives=2,reference_policy='retain_selected')
    np.testing.assert_array_equal(result['selected_positions'],[0,1,9,10,11])
    np.testing.assert_array_equal(result['training'],np.vstack((x,q[[0,1,9,10,11]])))
    np.testing.assert_array_equal(result['labels'],np.r_[y,[0,0,1,1,1]])
    np.testing.assert_array_equal(result['query'],q)
    result['query'][0,0]=1000
    assert q[0,0]!=1000


def test_real_branch_retraining_regenerates_each_expanded_population(monkeypatch):
    from sciona import santander_preparation as neural_prep
    from sciona import santander_tree as tree
    from sciona.santander_blend import blend_neural_tree
    x,y,q,s=sample();seen={}
    def capture(name,original):
        def wrapped(a,b,c):
            seen[name]=(a.copy(),b.copy(),c.copy())
            return original(a,b,c)
        return wrapped
    monkeypatch.setattr(neural_prep,'encode_populations',capture('neural',neural_prep.encode_populations))
    monkeypatch.setattr(tree,'encode_populations',capture('tree',tree.encode_populations))
    tree_params=dict(num_leaves=3,learning_rate=.1,feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,
        min_data_in_leaf=2,max_rounds=8,stopping_rounds=2,categorical=True)
    result=module.retrain(x,y,q,s,reference_policy='retain_selected',neural_positive=3,neural_negative=2,tree_positive=2,tree_negative=1,
        neural_controls=dict(folds=10,seeds=(42,),epochs=15,batch_size=1024),tree_controls=dict(folds=10,seeds=(42,),controls=tree_params))
    for name,indices,targets in [('neural',[0,1,9,10,11],[0,0,1,1,1]),('tree',[0,10,11],[0,1,1])]:
        a,b,c=seen[name]
        np.testing.assert_array_equal(a,np.vstack((x,q[indices])))
        np.testing.assert_array_equal(b,np.r_[y,targets]);np.testing.assert_array_equal(c,q)
        assert len(result['branches'][name]['models'])==10
    expected=blend_neural_tree(result['branches']['neural']['mean_ranks'],result['branches']['tree']['mean_ranks'])
    np.testing.assert_array_equal(result['scores'],expected)
    assert result['selection_counts']['neural']['labeled_rows']==45
    assert result['selection_counts']['tree']['labeled_rows']==43


def test_second_selection_failure_precedes_any_training(monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Training started')
    monkeypatch.setattr(module,'train_neural_cv',forbidden)
    x,y,q,s=sample()
    with pytest.raises(ValueError):module.retrain(x,y,q,s,reference_policy='retain_selected',
        neural_positive=2,neural_negative=1,tree_positive=12,tree_negative=1,neural_controls={},tree_controls={})


@pytest.mark.parametrize('problem',['policy','misaligned_scores','ties'])
def test_invalid_pseudo_population(problem):
    x,y,q,s=sample();policy='retain_selected'
    if problem=='policy':policy='unspecified'
    elif problem=='misaligned_scores':s=s[:-1]
    else:s[:]=0
    with pytest.raises(ValueError):module.expanded_population(x,y,q,s,positives=2,negatives=1,reference_policy=policy)

import numpy as np
import pytest
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold
from sciona import santander_tree as module


def test_augmentation_copy_counts_class_support_and_triplet_pairing():
    y=np.array([0,0,0,1,1]);x=np.arange(15,dtype=float).reshape(5,3)
    cat=(x.astype(int)%5);sub=x+100
    a,b,c,d=module.augment_reference(cat,x,sub,y,seed=7)
    assert len(d)==3*5+2*17 and np.sum(d==1)==34
    np.testing.assert_array_equal(c,b+100)
    np.testing.assert_array_equal(a,b.astype(int)%5)
    np.testing.assert_array_equal(b[:5],x)
    for label in (0,1):
        for column in range(3):assert set(b[d==label,column])==set(x[y==label,column])
    repeated=module.augment_reference(cat,x,sub,y,seed=7)
    for first,second in zip((a,b,c,d),repeated):np.testing.assert_array_equal(first,second)


def test_real_ten_fold_tree_branch_and_reference_only_augmentation(monkeypatch):
    rng=np.random.default_rng(4);y=np.tile([0,1],100)
    x=np.round(y[:,None]*10+rng.normal(0,.2,(200,4)),1)
    q=np.round(np.tile([0,1],6)[:,None]*10+rng.normal(0,.2,(12,4)),1)
    original=module.augment_reference;seen=[]
    def capture(a,b,c,d,**kwargs):
        seen.append((b.copy(),d.copy()));return original(a,b,c,d,**kwargs)
    monkeypatch.setattr(module,'augment_reference',capture)
    controls=dict(num_leaves=3,learning_rate=.1,feature_fraction=1.,bagging_fraction=1.,bagging_freq=0,
                  min_data_in_leaf=2,max_rounds=20,stopping_rounds=3,categorical=True)
    result=module.train_tree_cv(x,y,q,controls=controls)
    assert len(result['models'])==10 and result['model_probabilities'].shape==(10,12)
    splits=list(StratifiedKFold(10,shuffle=True,random_state=42).split(x,y))
    for (a,b),(fit,valid),record in zip(seen,splits,result['models']):
        np.testing.assert_array_equal(a,x[fit]);np.testing.assert_array_equal(b,y[fit])
        assert record['fit_rows']==180 and record['validation_rows']==20
        assert record['augmented_rows']==90*17+90*5
        assert record['restored_auc']>=.95
        assert record['best_iteration']==np.argmax(record['validation_auc'])+1
    oracle=np.mean([rankdata(p,method='average') for p in result['model_probabilities']],axis=0)
    np.testing.assert_array_equal(result['mean_ranks'],oracle)
    assert np.mean(oracle[1::2])>np.mean(oracle[::2])


@pytest.mark.parametrize('controls',[{},dict(categorical='yes'),dict(num_leaves=True)])
def test_incomplete_controls_rejected_before_encoding(controls,monkeypatch):
    def forbidden(*args,**kwargs):raise AssertionError('Encoding started')
    monkeypatch.setattr(module,'encode_populations',forbidden)
    with pytest.raises(ValueError):module.train_tree_cv([],[],[],controls=controls)

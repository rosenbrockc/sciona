import numpy as np
import pytest
from sciona import otto_meta_adaboost as module

CONTROLS=dict(algorithm='SAMME',boost_rounds=4,learning_rate=.7,trees=3,max_depth=3,min_samples_leaf=2,max_features=9)


def inputs():
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),40)
    x=np.eye(9)[y]*10+rng.uniform(0,.2,size=(360,9))
    noisy=y.copy();noisy[::7]=(noisy[::7]+1)%9
    return x,noisy,np.eye(9)*10+.1


def test_all_250_meta_fits_use_boosting_and_learn(monkeypatch):
    original=module.AdaBoostClassifier.fit;seen=[]
    def capture(self,x,y,*args,**kwargs):
        result=original(self,x,y,*args,**kwargs)
        seen.append((self.random_state,len(self.estimators_),self.estimator_errors_.copy()))
        return result
    monkeypatch.setattr(module.AdaBoostClassifier,'fit',capture)
    x,y,q=inputs();bag=module.fit_bag(x,y,q,seed=12,controls=CONTROLS)
    assert bag.shape==(250,9,9)
    assert [item[0] for item in seen]==list(range(12,262))
    assert all(count==4 for _,count,_ in seen)
    assert any(np.ptp(errors)>0 for _,_,errors in seen)
    np.testing.assert_array_equal(bag.mean(axis=0).argmax(axis=1),np.arange(9))


def test_query_reordering_preserves_seeded_bag():
    x,y,q=inputs()
    a=module.fit_bag(x,y,q,seed=2**32-2,controls=CONTROLS)
    b=module.fit_bag(x,y,q[::-1],seed=2**32-2,controls=CONTROLS)
    np.testing.assert_array_equal(a,b[:,::-1,:])


def test_unavailable_historical_algorithm_is_not_silently_replaced():
    x,y,q=inputs();options=dict(CONTROLS,algorithm='SAMME.R')
    with pytest.raises(ValueError,match='SAMME'):module.fit_bag(x,y,q,seed=12,controls=options)

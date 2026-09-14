import numpy as np
import pytest
from sciona import otto_tsne as module


def inputs():
    rng=np.random.default_rng(12)
    return rng.integers(0,12,size=(72,8)).astype(float),[f'r{i}' for i in range(72)]


def fit(x,ids):return module.fit_population(x,ids,seed=12,perplexity=8,learning_rate=20,max_iter=350)


def test_actual_three_dimensional_embedding_repeats_and_preserves_inputs():
    x,ids=inputs();before=x.copy();a=fit(x,ids);b=fit(x,ids)
    assert a.coordinates.shape==(72,3) and np.isfinite(a.kl_divergence)
    np.testing.assert_array_equal(a.coordinates,b.coordinates)
    np.testing.assert_array_equal(x,before)
    assert not a.coordinates.flags.writeable
    assert np.std(a.coordinates)>0


def test_identity_lookup_reorders_without_refitting_and_returns_copy():
    x,ids=inputs();model=fit(x,ids)
    actual=model.lookup([ids[5],ids[0]])
    np.testing.assert_array_equal(actual,model.coordinates[[5,0]])
    actual[:]=0
    assert np.any(model.coordinates[[5,0]]!=0)
    with pytest.raises(ValueError,match='outside fitted population'):model.lookup(['new-row'])


def test_entire_declared_population_is_log_transformed(monkeypatch):
    x,ids=inputs();original=module.TSNE.fit_transform;seen=[]
    def capture(self,values,*args,**kwargs):
        seen.append(values.copy());return original(self,values,*args,**kwargs)
    monkeypatch.setattr(module.TSNE,'fit_transform',capture)
    fit(x,ids)
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0],np.log1p(x))


@pytest.mark.parametrize('problem',['duplicate','perplexity','iterations','negative'])
def test_invalid_population_and_controls(problem):
    x,ids=inputs();kwargs=dict(seed=12,perplexity=8,learning_rate=20,max_iter=350)
    if problem=='duplicate':ids[1]=ids[0]
    elif problem=='perplexity':kwargs['perplexity']=len(x)
    elif problem=='iterations':kwargs['max_iter']=250
    else:x[0,0]=-1
    with pytest.raises(ValueError):module.fit_population(x,ids,**kwargs)

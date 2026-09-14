import numpy as np
import pytest
from sciona.otto_tsne import fit_population
from sciona import otto_embedding_boosting as module
from sciona.otto_embedding_crossfit import crossfit_boosting


@pytest.fixture(scope='module')
def population():
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),20)
    x=np.eye(9)[y]*20+rng.uniform(0,.2,size=(180,9));q=np.array([x[y==i].mean(axis=0) for i in range(9)])
    ids=[f'r{i}' for i in range(180)];qids=[f'q{i}' for i in range(9)]
    e=fit_population(np.vstack((x,q)),ids+qids,seed=12,perplexity=10,learning_rate=20,max_iter=350)
    return x,y,np.arange(180)%5,ids,q,qids,e


def run(population,variant):
    *args,embedding=population
    return crossfit_boosting(*args,embedding=embedding,variant=variant,seed=12,controls={'clustering':dict(cluster_counts=[2,3],ddof=1,n_init=2,max_iter=100),'boosting':dict(rounds=8,max_depth=3,eta=.4,subsample=1.,colsample_bytree=1.)})


@pytest.mark.parametrize('variant',['raw','log1p','scaled_raw'])
def test_actual_embedding_fold_local_cluster_fit_and_learning(population,variant,monkeypatch):
    original=module.fit_clusters;seen=[];encoded=[];fit_encoded=module._fit_encoded
    def capture(x,**kwargs):
        seen.append((x.copy(),kwargs));return original(x,**kwargs)
    monkeypatch.setattr(module,'fit_clusters',capture)
    def capture_encoded(x,y,q,**kwargs):
        encoded.append((x.copy(),q.copy()));return fit_encoded(x,y,q,**kwargs)
    monkeypatch.setattr(module,'_fit_encoded',capture_encoded)
    result=run(population,variant)
    x,y,f,ids,q,qids,e=population
    assert len(seen)==6 and result['model_fits']==6
    for fold in range(5):np.testing.assert_array_equal(seen[fold][0],x[f!=fold])
    np.testing.assert_array_equal(seen[-1][0],x)
    assert encoded[-1][0].shape==(len(x),x.shape[1]+5)
    np.testing.assert_array_equal(encoded[-1][0][:,:x.shape[1]],x)
    np.testing.assert_array_equal(encoded[-1][0][:,x.shape[1]:x.shape[1]+3],e.lookup(ids))
    np.testing.assert_array_equal(encoded[-1][1][:,x.shape[1]:x.shape[1]+3],e.lookup(qids))
    assert all(k['standardize']==(variant=='scaled_raw') and k['kind']==('log1p' if variant=='log1p' else 'raw') for _,k in seen)
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))
    np.testing.assert_allclose(result['oof'].sum(axis=1),1.,atol=1e-6)


@pytest.mark.parametrize('variant',['raw','log1p','scaled_raw'])
def test_heldout_labels_and_query_values_isolated_with_fixed_embedding(population,variant):
    baseline=run(population,variant);x,y,f,ids,q,qids,e=population
    changed=y.copy();held=f==0;changed[held]=(changed[held]+1)%9
    other=run((x,changed,f,ids,q,qids,e),variant)
    np.testing.assert_array_equal(baseline['oof'][held],other['oof'][held])
    other=run((x,y,f,ids,q*2,qids,e),variant)
    np.testing.assert_array_equal(baseline['oof'],other['oof'])


def test_unknown_embedding_identity_rejected(population):
    x,y,f,ids,q,qids,e=population
    with pytest.raises(ValueError,match='outside fitted population'):
        run((x,y,f,ids,q,['unknown']+qids[1:],e),'raw')

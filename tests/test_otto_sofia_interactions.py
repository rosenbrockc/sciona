from pathlib import Path
import numpy as np
import pytest
from sciona.otto_tsne import fit_population
from sciona.otto_interactions import InteractionSelection
from sciona import otto_sofia_interactions as module
from sciona.otto_sofia_interactions_crossfit import crossfit_sofia

CONTROLS=dict(forest=dict(ntree=32,mtry=4,nodesize=1),triples=[(0,1,2),(3,4,5)],ddof=1,r_library=str(Path(__file__).resolve().parents[1]/'.venv/otto-r-library'),sofia=dict(regularization=.01,iterations=10000))


@pytest.fixture(scope='module')
def population():
    rng=np.random.default_rng(12);y=np.tile(np.arange(9),20)
    x=np.column_stack((np.eye(9)[y]*20,np.ones((180,4))))+rng.uniform(0,.1,size=(180,13))
    q=np.array([x[y==i].mean(axis=0) for i in range(9)])
    ids=[f'r{i}' for i in range(180)];qids=[f'q{i}' for i in range(9)]
    e=fit_population(np.vstack((x,q)),ids+qids,seed=12,perplexity=10,learning_rate=20,max_iter=350)
    return x,y,np.arange(180)%5,ids,q,qids,e


def run(p):
    *args,e=p
    return crossfit_sofia(*args,embedding=e,seed=12,controls=CONTROLS)


def test_assembly_matches_independent_reference_statistics(population,monkeypatch):
    x,y,f,ids,q,qids,e=population
    monkeypatch.setattr(module,'fit_selection',lambda *a,**k:InteractionSelection(13,tuple(range(13))))
    a,b=module.assemble(x,y,ids,q,qids,embedding=e,seed=12,**{k:v for k,v in CONTROLS.items() if k!='sofia'})
    def encoded(x,ids):return np.column_stack((x,e.lookup(ids),x[:,0]*x[:,1]*x[:,2],x[:,3]*x[:,4]*x[:,5]))
    reference=encoded(x,ids);query=encoded(q,qids)
    mean=reference.mean(axis=0);std=reference.std(axis=0,ddof=1)
    np.testing.assert_allclose(a,(reference-mean)/std)
    np.testing.assert_allclose(b,(query-mean)/std)


def test_actual_fold_local_selection_and_native_learning(population,monkeypatch):
    original=module.fit_selection;seen=[]
    def capture(x,y,**kwargs):
        seen.append((x.copy(),y.copy()));return original(x,y,**kwargs)
    monkeypatch.setattr(module,'fit_selection',capture)
    result=run(population);x,y,f,*_=population
    assert len(seen)==6 and result['binary_models']==54
    for fold in range(5):
        np.testing.assert_array_equal(seen[fold][0],x[f!=fold])
        np.testing.assert_array_equal(seen[fold][1],y[f!=fold])
    np.testing.assert_array_equal(result['query'].argmax(axis=1),np.arange(9))


def test_heldout_labels_and_query_isolation_given_fixed_embedding(population):
    baseline=run(population);x,y,f,ids,q,qids,e=population
    changed=y.copy();held=f==0;changed[held]=(changed[held]+1)%9
    np.testing.assert_array_equal(baseline['oof'][held],run((x,changed,f,ids,q,qids,e))['oof'][held])
    np.testing.assert_array_equal(baseline['oof'],run((x,y,f,ids,q*2,qids,e))['oof'])

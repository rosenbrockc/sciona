import numpy as np
import pytest
from sciona.otto_tsne import fit_population
from sciona import otto_supplemental as module


@pytest.fixture(scope='module')
def population():
    rng=np.random.default_rng(12);x=rng.integers(0,12,size=(90,13)).astype(float);y=np.arange(90)%9;f=np.arange(90)%5
    q=rng.integers(0,12,size=(2,13)).astype(float);ids=[f'r{i}' for i in range(90)];qids=['q0','q1']
    e=fit_population(np.vstack((x,q)),ids+qids,seed=12,perplexity=8,learning_rate=20,max_iter=350)
    return x,y,f,ids,q,qids,e


def run(p):
    *args,e=p
    return module.build(*args,embedding=e,seed=12,raw_metrics=['euclidean','cityblock','braycurtis'],tfidf_metrics=['euclidean'],embedding_metrics=['euclidean','cityblock'],tfidf_controls=dict(smooth_idf=True,sublinear_tf=False,norm='l2'),cluster_controls=dict(kind='raw',standardize=False,ddof=1,cluster_counts=[2,3],n_init=2,max_iter=100))


def test_all_blocks_execute_and_distance_oracles(population):
    result=run(population);x,y,f,ids,q,qids,e=population;blocks=result['supplemental']
    assert set(blocks)==set(range(1,8))
    assert [blocks[i].training.shape[1] for i in range(1,8)]==[27,27,27,9,18,2,1]
    for row in (0,17):
        for label in range(9):
            reference=x[(f!=f[row])&(y==label)]
            distances=sorted(float(np.linalg.norm(x[row]-r)) for r in reference)
            for entry,k in ((1,1),(2,2),(3,4)):
                np.testing.assert_allclose(blocks[entry].training[row,label],sum(distances[:k]))
            z=e.lookup(ids);r=z[(f!=f[row])&(y==label)]
            np.testing.assert_allclose(blocks[5].training[row,label],min(float(np.linalg.norm(z[row]-v)) for v in r),rtol=1e-6)
    np.testing.assert_array_equal(blocks[7].training[:,0],np.count_nonzero(x,axis=1))
    np.testing.assert_array_equal(result['raw_neural'].training,x)
    for i in range(1,6):np.testing.assert_array_equal(blocks[i].folds,f)


def test_own_fold_labels_do_not_change_distance_features(population):
    a=run(population);x,y,f,ids,q,qids,e=population;changed=y.copy();held=f==0;changed[held]=(changed[held]+1)%9
    b=run((x,changed,f,ids,q,qids,e))
    for entry in range(1,6):np.testing.assert_array_equal(a['supplemental'][entry].training[held],b['supplemental'][entry].training[held])


def test_clusters_fit_only_full_training(population,monkeypatch):
    original=module.fit_clusters;seen=[]
    def capture(x,**kwargs):seen.append(x.copy());return original(x,**kwargs)
    monkeypatch.setattr(module,'fit_clusters',capture)
    run(population)
    assert len(seen)==1
    np.testing.assert_array_equal(seen[0],population[0])

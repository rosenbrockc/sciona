import numpy as np
import pytest
from sciona.otto_tsne import fit_population,PopulationEmbedding
from sciona import otto_tsne_clusters as module


def fit(embedding,ids):
    return module.fit_embedding_clusters(embedding,ids,cluster_counts=[2,3],seed=12,n_init=4,max_iter=100)


def synthetic_embedding():
    coordinates=np.array([[-10,0,1],[-9,0,1],[0,0,-1],[1,0,-1],[9,0,2],[10,0,2],[100,0,0]],dtype=float)
    return PopulationEmbedding(tuple(f'r{i}' for i in range(7)),coordinates,0.)


def test_signed_coordinates_and_hand_nearest_center_oracle():
    e=synthetic_embedding();model=fit(e,list(e.identities[:6]))
    output=model.lookup(list(e.identities))
    assert output.shape==(7,5)
    np.testing.assert_array_equal(output[:,:3],e.coordinates)
    for col,centers in enumerate(model.centers):
        for row,x in enumerate(e.coordinates):
            expected=min(range(len(centers)),key=lambda i:sum(float(v)**2 for v in x-centers[i]))
            assert output[row,3+col]==expected
        assert not centers.flags.writeable


def test_reference_population_and_query_lookup_do_not_refit(monkeypatch):
    e=synthetic_embedding();original=module.KMeans.fit;seen=[]
    def capture(self,x,*args,**kwargs):
        seen.append(x.copy());return original(self,x,*args,**kwargs)
    monkeypatch.setattr(module.KMeans,'fit',capture)
    model=fit(e,list(e.identities[:6]));baseline=model.lookup(['r0'])
    model.lookup(['r6','r0'])
    assert len(seen)==2
    for x in seen:np.testing.assert_array_equal(x,e.coordinates[:6])
    np.testing.assert_array_equal(baseline,model.lookup(['r0']))
    with pytest.raises(ValueError,match='outside fitted population'):model.lookup(['unknown'])


def test_actual_tsne_and_two_cluster_feature_assembly():
    rng=np.random.default_rng(12);x=rng.integers(0,12,size=(72,8)).astype(float);ids=[f'r{i}' for i in range(72)]
    embedding=fit_population(x,ids,seed=12,perplexity=8,learning_rate=20,max_iter=350)
    a=fit(embedding,ids[:60]);b=fit(embedding,ids[:60])
    np.testing.assert_array_equal(a.lookup(ids),b.lookup(ids))
    assert a.lookup(ids[60:]).shape==(12,5)


@pytest.mark.parametrize('counts',[[2],[2,2],[2,3,4],[2,9]])
def test_invalid_cluster_counts(counts):
    e=synthetic_embedding()
    with pytest.raises(ValueError):module.fit_embedding_clusters(e,list(e.identities),cluster_counts=counts,seed=12,n_init=4,max_iter=100)

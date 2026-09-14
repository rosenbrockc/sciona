import numpy as np
import pytest
from sciona.otto_augmented_features import fit_augmentation


def fit():
    reference=np.column_stack((np.arange(16),np.arange(16)[::-1]))
    result=fit_augmentation(reference,cluster_counts=list(range(2,9)),ddof=1,seed=12,n_init=2,max_iter=100)
    return reference,result


def test_seven_clusters_and_hand_row_statistics():
    reference,model=fit();query=np.array([[0,0],[15,15],[7,8]])
    result=model.transform(query)
    assert result.shape==(3,12)
    np.testing.assert_array_equal(result[:,:2],query)
    np.testing.assert_array_equal(result[:,-3], [2,0,0])
    mean=reference.mean(axis=0);sd=np.sqrt(((reference-mean)**2).sum(axis=0)/15)
    z=(query-mean)/sd
    np.testing.assert_array_equal(result[:,-2],(z>.5).sum(axis=1))
    np.testing.assert_array_equal(result[:,-1],(z<-.5).sum(axis=1))
    for col,cluster in enumerate(model.clusters.models):
        distances=((query[:,None,:]-cluster.cluster_centers_[None,:,:])**2).sum(axis=2)
        np.testing.assert_array_equal(result[:,2+col],distances.argmin(axis=1))


def test_fit_population_and_query_independence():
    reference,model=fit();a=model.transform([[3,4]])
    before=model.scaling.mean.copy();centers=[m.cluster_centers_.copy() for m in model.clusters.models]
    reference[:]=999
    b=model.transform([[3,4],[999,999]])
    np.testing.assert_array_equal(a,b[:1]);np.testing.assert_array_equal(model.scaling.mean,before)
    for m,c in zip(model.clusters.models,centers):np.testing.assert_array_equal(m.cluster_centers_,c)


def test_seeded_repeat():
    reference,a=fit();_,b=fit()
    np.testing.assert_array_equal(a.transform(reference),b.transform(reference))


@pytest.mark.parametrize('counts',[list(range(2,8)),[2]*7,list(range(20,27))])
def test_invalid_cluster_bank(counts):
    with pytest.raises(ValueError):fit_augmentation([[i,i+1] for i in range(16)],cluster_counts=counts,ddof=1,seed=12,n_init=2,max_iter=100)

import copy
import numpy as np
import pytest
from sciona.otto_clustering import fit_clusters


def fit(x,**overrides):
    c=dict(kind='raw',standardize=False,ddof=0,cluster_counts=[2],seed=17,n_init=5,max_iter=100)
    c.update(overrides)
    return fit_clusters(x,**c)


def test_hand_centers_distances_and_assignments():
    bank=fit([[0,0],[0,2],[10,0],[10,2]])
    centers=bank.models[0].cluster_centers_
    np.testing.assert_allclose(centers[np.argsort(centers[:,0])],[[0,1],[10,1]])
    q=np.array([[0,1],[5,1],[10,1]])
    result=bank.transform(q)
    expected=np.array([[np.sqrt(sum((point-center)**2)) for center in centers] for point in q])
    np.testing.assert_allclose(result['distances'],expected)
    np.testing.assert_array_equal(result['assignments'][:,0],expected.argmin(axis=1))


def test_reference_only_scaling_and_detached_fit():
    reference=np.array([[1.,1],[1,2],[20,1],[20,2]])
    bank=fit(reference,kind='log1p',standardize=True)
    means=bank.scaling.mean.copy();centers=bank.models[0].cluster_centers_.copy()
    first=bank.transform([[2,1]])
    reference[:]=999
    second=bank.transform([[2,1],[1000,1000]])
    np.testing.assert_array_equal(first['distances'],second['distances'][:1])
    np.testing.assert_array_equal(bank.scaling.mean,means)
    np.testing.assert_array_equal(bank.models[0].cluster_centers_,centers)


def test_multiple_cluster_counts_and_repeat():
    x=np.arange(24).reshape(12,2)
    a=fit(x,cluster_counts=[2,3]);b=fit(x,cluster_counts=[2,3])
    first=a.transform(x);second=b.transform(x)
    assert first['distances'].shape==(12,5) and first['assignments'].shape==(12,2)
    for key in first:np.testing.assert_array_equal(first[key],second[key])


@pytest.mark.parametrize('problem',['too_many','duplicates','bad_seed','degenerate','negative'])
def test_invalid_fitting_controls(problem):
    x=[[0.,0],[1,1],[2,2]];c={}
    if problem=='too_many':c['cluster_counts']=[4]
    if problem=='duplicates':c['cluster_counts']=[2,2]
    if problem=='bad_seed':c['seed']=True
    if problem=='degenerate':x=[[1,1]]*3
    if problem=='negative':x[0][0]=-1
    with pytest.raises(ValueError):fit(x,**c)

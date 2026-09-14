import numpy as np
import pytest
from sciona.otto_neighbor_features import class_distances


def fixture():
    x=np.array([[c+1.,i+1.] for c in range(9) for i in range(4)])
    return x,np.repeat(np.arange(9),4),np.array([[1.,1.],[3.,2.]])


@pytest.mark.parametrize('metric',['euclidean','cityblock','braycurtis'])
def test_hand_pairwise_class_distances(metric):
    x,y,q=fixture();actual=class_distances(x,y,q,metric=metric,chunk_size=1)
    expected=np.empty((2,3,9))
    for row,point in enumerate(q):
        for label in range(9):
            distances=[]
            for ref in x[y==label]:
                delta=abs(point-ref)
                d=float(np.sqrt(sum(delta**2))) if metric=='euclidean' else float(sum(delta))
                if metric=='braycurtis':d/=sum(point+ref)
                distances.append(d)
            for col,k in enumerate((1,2,4)):expected[row,col,label]=sum(sorted(distances)[:k])
    np.testing.assert_allclose(actual,expected)


def test_query_chunk_and_population_independence():
    x,y,q=fixture()
    a=class_distances(x,y,q,metric='cityblock',chunk_size=1)
    b=class_distances(x,y,q,metric='cityblock',chunk_size=128)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[:1],class_distances(x,y,q[:1],metric='cityblock'))


def test_reference_order_invariance():
    x,y,q=fixture();order=np.random.default_rng(4).permutation(len(x))
    np.testing.assert_array_equal(class_distances(x,y,q,metric='euclidean'),class_distances(x[order],y[order],q,metric='euclidean'))


def test_zero_bray_curtis_convention():
    x,y,q=fixture();x[:]=0;q[:]=0
    np.testing.assert_array_equal(class_distances(x,y,q,metric='braycurtis'),0.)


@pytest.mark.parametrize('problem',['missing_reference','bad_labels','nan','negative_bray'])
def test_invalid_reference_population(problem):
    x,y,q=fixture()
    if problem=='missing_reference':x=x[:-1];y=y[:-1]
    if problem=='bad_labels':y=y.astype(float)
    if problem=='nan':q[0,0]=np.nan
    if problem=='negative_bray':q[0,0]=-1
    with pytest.raises(ValueError):class_distances(x,y,q,metric='braycurtis')

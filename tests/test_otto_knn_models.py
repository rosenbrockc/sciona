import numpy as np
import pytest
from sciona.otto_knn_models import predict,crossfit,NEIGHBORS


def fixture():
    x=np.arange(1440,dtype=float)[:,None]+1
    y=np.arange(1440)%9;f=np.arange(1440)%5
    return x,y,f,[f'synthetic-{i}' for i in range(1440)],np.array([[.2],[1500.]]),['synthetic-q0','synthetic-q1']


def test_all_ten_neighbor_counts_match_direct_sorted_labels():
    x,y,_,_,q,_=fixture()
    result=predict(x,y,q,metric='euclidean')
    for row,point in enumerate(q):
        ordered=sorted(range(len(x)),key=lambda i:abs(float(x[i,0]-point[0])))
        for col,k in enumerate(NEIGHBORS):
            expected=np.bincount(y[ordered[:k]],minlength=9)/k
            np.testing.assert_array_equal(result[row,col],expected)
    assert np.count_nonzero(result[0,0])==2
    np.testing.assert_allclose(result.sum(axis=2),1.)


def test_equal_distance_ties_and_chunk_independence():
    x,y,_,_,q,_=fixture();x[:]=1;q[:]=1
    a=predict(x,y,q,metric='cityblock',chunk_size=1)
    b=predict(x,y,q,metric='cityblock',chunk_size=128)
    np.testing.assert_array_equal(a,b)
    np.testing.assert_array_equal(a[0,0],np.bincount(y[:2],minlength=9)/2)


def test_five_fold_exclusion_and_full_query_reference():
    x,y,f,ids,q,qids=fixture()
    result=crossfit(x,y,f,ids,q,qids,metric='cityblock')
    assert result['oof'].shape==(1440,10,9)
    for i in [0,101,1439]:
        candidates=[j for j in range(len(x)) if f[j]!=f[i]]
        ordered=sorted(candidates,key=lambda j:abs(float(x[j,0]-x[i,0])))
        for col,k in enumerate(NEIGHBORS):
            np.testing.assert_array_equal(result['oof'][i,col],np.bincount(y[ordered[:k]],minlength=9)/k)
    np.testing.assert_array_equal(result['query'],predict(x,y,q,metric='cityblock'))


def test_own_fold_labels_and_query_do_not_leak():
    x,y,f,ids,q,qids=fixture()
    a=crossfit(x,y,f,ids,q,qids,metric='euclidean')
    altered=y.copy();altered[f==0]=(altered[f==0]+1)%9
    b=crossfit(x,altered,f,ids,q*2,qids,metric='euclidean')
    np.testing.assert_array_equal(a['oof'][f==0],b['oof'][f==0])


@pytest.mark.parametrize('problem',['short_reference','overlap','short_fold'])
def test_invalid_reference_requirements(problem):
    x,y,f,ids,q,qids=fixture()
    with pytest.raises(ValueError):
        if problem=='short_reference':predict(x[:1023],y[:1023],q,metric='euclidean')
        elif problem=='overlap':crossfit(x,y,f,ids,q,ids[:2],metric='euclidean')
        else:crossfit(x[:1200],y[:1200],f[:1200],ids[:1200],q,qids,metric='euclidean')

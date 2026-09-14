import numpy as np
import pytest
from scipy.sparse import vstack
from sciona.temporal_sparse_features import TemporalSparseEncoder


def events():
    return [dict(entity='a',time=t,value=1.,category='x',target=y) for t,y in [(0.,2.),(0.,4.),(1.,8.),(2.,10.)]]


def encode(chunk):
    encoder=TemporalSparseEncoder(hash_size=16,lookback=10.,period=4.)
    batches=list(encoder.encode(iter(events()),chunk_size=chunk,labeled=True))
    return vstack([x for x,_ in batches]).toarray(),np.concatenate([y for _,y in batches])


def test_chunk_boundaries_do_not_reveal_same_time_targets():
    one,labels=encode(1);three,_=encode(3)
    np.testing.assert_array_equal(one,three)
    np.testing.assert_array_equal(one[:2,3:8],np.zeros((2,5)))
    np.testing.assert_allclose(one[2,3:8],[3.,3.,1.,2.,1.])
    np.testing.assert_allclose(one[3,3:8],[14/3,8.,1.,3.,1.])
    np.testing.assert_array_equal(labels,[2.,4.,8.,10.])
    assert one.shape==(4,24) and np.all(one[:,8:].sum(1)==2)


def test_future_labels_do_not_affect_prior_features():
    first,_=encode(2);changed=events();changed[-1]['target']=999.
    encoder=TemporalSparseEncoder(hash_size=16,lookback=10.,period=4.)
    actual=vstack([x for x,_ in encoder.encode(changed,chunk_size=2,labeled=True)]).toarray()
    np.testing.assert_array_equal(first,actual)


def test_history_expiration_and_unlabeled_query():
    encoder=TemporalSparseEncoder(hash_size=16,lookback=1.,period=4.)
    list(encoder.encode(events()[:2],chunk_size=2,labeled=True))
    query=[dict(entity='a',time=3.,value=1.,category='new')]
    [(x,y)]=list(encoder.encode(query,chunk_size=2,labeled=False))
    assert y is None
    np.testing.assert_array_equal(x.toarray()[0,3:8],np.zeros(5))


def test_generator_consumption_bounded_by_chunk():
    consumed=[]
    def source():
        for e in events():consumed.append(e);yield e
    iterator=TemporalSparseEncoder().encode(source(),chunk_size=2,labeled=True)
    next(iterator)
    assert len(consumed)==2

@pytest.mark.parametrize('kwargs',[dict(max_pending=1),dict(max_history=1)])
def test_capacity_failure(kwargs):
    with pytest.raises(ValueError):list(TemporalSparseEncoder(**kwargs).encode(events(),chunk_size=1,labeled=True))


def test_entity_capacity_and_time_order():
    data=events();data[1]['entity']='b'
    with pytest.raises(ValueError):list(TemporalSparseEncoder(max_entities=1).encode(data,chunk_size=1,labeled=True))
    with pytest.raises(ValueError):list(TemporalSparseEncoder().encode(events()[::-1],chunk_size=1,labeled=True))


def test_same_time_order_does_not_change_later_lags():
    first,_=encode(2);data=events();data[:2]=data[:2][::-1]
    encoder=TemporalSparseEncoder(hash_size=16,lookback=10.,period=4.)
    changed=vstack([x for x,_ in encoder.encode(data,chunk_size=1,labeled=True)]).toarray()
    np.testing.assert_array_equal(first[2:],changed[2:])

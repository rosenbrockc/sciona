import numpy as np
import torch

from sciona.champs_population import split_population,population_batches


def synthetic():
    return tuple(torch.arange(10).reshape(10,1)+i*100 for i in range(10))


def test_source_split_and_alignment():
    packed=synthetic()
    train,validation=split_population(packed)
    assert train[0][:,0].tolist()==[2,8,4,9,1,6,0,5]
    assert validation[0][:,0].tolist()==[7,3]
    for i in range(10):
        assert torch.equal(train[i]-i*100,train[0])
        assert torch.equal(validation[i]-i*100,validation[0])
    assert set(train[0][:,0].tolist()).isdisjoint(validation[0][:,0].tolist())


def test_batch_shuffle_repeatable_without_global_rng_change():
    ts=torch.random.get_rng_state().clone()
    ns=np.random.get_state()
    def batches():
        return list(population_batches(synthetic(),batch_size=4,shuffle=True,drop_last=False,seed=42))
    first,second=batches(),batches()
    assert [b[0].shape[0] for b in first]==[4,4,2]
    for a,b in zip(first,second):
        assert torch.equal(a[0],b[0])
    assert sorted(torch.cat([b[0] for b in first])[:,0].tolist())==list(range(10))
    assert torch.equal(ts,torch.random.get_rng_state())
    current=np.random.get_state()
    np.testing.assert_array_equal(ns[1],current[1])
    assert ns[0]==current[0] and ns[2:]==current[2:]


def test_source_drop_last_behavior():
    batches=list(population_batches(synthetic(),batch_size=3,shuffle=False,drop_last=True,seed=0))
    assert [b[0].shape[0] for b in batches]==[3,3,3]
    assert torch.cat([b[0] for b in batches])[:,0].tolist()==list(range(9))

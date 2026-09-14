"""Synthetic population-level weighting and multimodal alignment checks."""
import numpy as np
import pytest
from sciona.openvaccine_population import population_weights,reverse_training_batch


def test_weights_count_full_population_and_apply_proximity():
    ids=np.array([1,2,2,2,2,3],dtype=np.int64)
    proximity=np.array([1,1,2,3,4,0],dtype=np.float32)
    weights=population_weights(ids,proximity)
    np.testing.assert_array_equal(weights,[1,.5,1,1.5,2,0])
    order=np.array([5,3,0,2,4,1])
    np.testing.assert_array_equal(population_weights(ids[order],proximity[order]),weights[order])
    np.testing.assert_array_equal(ids,[1,2,2,2,2,3])
    np.testing.assert_array_equal(proximity,[1,1,2,3,4,0])


def test_reverse_keeps_all_modalities_aligned_and_roundtrips():
    nodes=np.arange(2*6*55,dtype=np.float32).reshape(2,6,55)
    adjacency=np.arange(2*6*6*8,dtype=np.float32).reshape(2,6,6,8)
    targets=np.arange(2*4*5,dtype=np.float32).reshape(2,4,5);targets[0,0,2]=np.nan
    weights=np.array([1,.5],dtype=np.float32)
    flags=np.array([True,False])
    originals=[x.copy() for x in [nodes,adjacency,targets,weights]]
    result=reverse_training_batch(nodes,adjacency,targets,weights,flags)
    for pos in range(6):
        np.testing.assert_array_equal(result[0][0,pos],nodes[0,5-pos])
        for other in range(6):np.testing.assert_array_equal(result[1][0,pos,other],adjacency[0,5-pos,5-other])
    for pos in range(4):np.testing.assert_array_equal(result[2][0,pos],targets[0,3-pos])
    for changed,original in zip(result,originals):np.testing.assert_array_equal(changed[1],original[1])
    restored=reverse_training_batch(*result,flags)
    for value,original in zip(restored,originals):np.testing.assert_array_equal(value,original)
    for value,original in zip([nodes,adjacency,targets,weights],originals):np.testing.assert_array_equal(value,original)


@pytest.mark.parametrize('ids,factors', [([],[]),([1],[0]),([1],[-1]),([1],[float('nan')]),([1,2],[1])])
def test_invalid_population(ids,factors):
    with pytest.raises(ValueError):population_weights(np.array(ids,dtype=np.int64),np.array(factors,dtype=np.float32))


@pytest.mark.parametrize('bad',['flags','weight','targets'])
def test_invalid_reversal(bad):
    nodes=np.ones((1,4,55),dtype=np.float32);adj=np.ones((1,4,4,8),dtype=np.float32)
    target=np.ones((1,2,5),dtype=np.float32);weight=np.ones(1,dtype=np.float32);flag=np.ones(1,dtype=bool)
    if bad=='flags':flag=flag.astype(np.int64)
    if bad=='weight':weight[:]=0
    if bad=='targets':target[0,0,0]=np.inf
    with pytest.raises(ValueError):reverse_training_batch(nodes,adj,target,weight,flag)

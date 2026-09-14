"""Synthetic full block ordering and unbinned prediction-slot routing."""
import numpy as np
import pytest
from sciona.amex_assembly import BLOCK_ORDER,assemble


def test_exact_block_order_missing_codes_and_prediction_routing():
    blocks={name:np.full((3,1),float(i)) for i,name in enumerate(reversed(BLOCK_ORDER))}
    blocks[BLOCK_ORDER[0]][0,0]=np.nan
    p=[[.2,.7],[.8],[.1,.4,.9]];r=assemble(blocks,p)
    assert r['tree_features'].shape==r['neural_features'].shape==(3,22)
    np.testing.assert_array_equal(r['tree_features'][:,:9],np.column_stack([blocks[n] for n in BLOCK_ORDER]))
    assert np.isnan(r['tree_features'][0,0]) and r['neural_features'][0,0]==0
    np.testing.assert_array_equal(r['neural_features'][0,-2:],[.2,.7])
    assert np.all(r['neural_features'][0,9:-2]==0)
    assert np.isnan(r['tree_features'][0,9:-2]).all()
    assert np.all(r['neural_features'][1,:9]==1)


def test_missing_block_and_prediction_alignment_rejected():
    blocks={name:np.ones((3,1)) for name in BLOCK_ORDER}
    with pytest.raises(ValueError):assemble(blocks,[[.5]])
    blocks.pop(BLOCK_ORDER[0])
    with pytest.raises(ValueError):assemble(blocks,[[.5]]*3)

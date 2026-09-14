"""Synthetic categorical aggregation and vocabulary oracles."""
import numpy as np
import pytest
from sciona.amex_categorical import summarize_populations


def test_joint_vocabulary_missing_rows_and_last():
    xs=[[[1.],[np.nan],[1.]],[[2.]]];masks=[[True,False,True],[True]]
    r=summarize_populations(xs,masks)
    np.testing.assert_array_equal(r['vocabulary'][0],[1,2])
    expected=[2/3,np.sqrt(1/3),2,1,0,0,0,0,1,1,2]
    np.testing.assert_allclose(r['values'][0],expected)
    assert np.isnan(r['values'][1,1]) and np.isnan(r['values'][1,5])
    assert r['values'][1,-1]==1


def test_all_missing_category_and_window_omission():
    xs=[[[np.nan,1.],[np.nan,2.]],[[np.nan,2.]]]
    r=summarize_populations(xs,[[True,True],[False]],windowed=True)
    assert len(r['vocabulary'][0])==0
    assert r['values'].shape==(2,9)
    np.testing.assert_allclose(r['values'][0],[.5,np.sqrt(.5),1,.5,np.sqrt(.5),1,0,2,2])
    assert r['values'][1,-1]==0


def test_category_order_and_population_permutation():
    xs=[[[4.,2.],[1.,3.]],[[2.,2.],[4.,2.]]];m=[[True,True]]*2
    a=summarize_populations(xs,m)['values']
    b=summarize_populations(xs[::-1],m)['values']
    np.testing.assert_allclose(a,b[::-1])


def test_invalid_masks_width_and_mode():
    with pytest.raises(ValueError):summarize_populations([[[1.]]],[[1]])
    with pytest.raises(ValueError):summarize_populations([[[1.]],[[1.,2.]]],[[True],[True]])
    with pytest.raises(ValueError):summarize_populations([[[1.]]],[[True]],windowed=1)

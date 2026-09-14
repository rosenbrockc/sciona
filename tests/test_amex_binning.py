"""Synthetic greedy boundary, missing-code and compressed-branch oracles."""
import numpy as np
import pytest
from sciona.amex_binning import boundaries,normalize


def test_uncompressed_count_threshold():
    np.testing.assert_array_equal(boundaries([0,1,2,3],[1,2,3,1]),[1.5,2.5,np.inf])
    np.testing.assert_array_equal(boundaries([7],[4]),[np.inf])


def test_compressed_last_split_omission():
    np.testing.assert_array_equal(boundaries(np.arange(8),np.ones(8,int)*3,max_bins=3),[2.5,np.inf])


def test_normalized_joint_codes_and_missing():
    x=np.array([[0.],[0.],[0.],[2.],[2.],[2.],[np.nan]])
    np.testing.assert_array_equal(normalize(x)['values'].ravel(),[.5,.5,.5,1,1,1,0])
    assert np.isnan(x[-1,0])


def test_undefined_missing_and_invalid_counts():
    with pytest.raises(ValueError):normalize([[np.nan]])
    with pytest.raises(ValueError):boundaries([1,2],[0,2])
    with pytest.raises(ValueError):boundaries([2,1],[1,1])
    with pytest.raises(ValueError):boundaries([1],[1],max_bins=True)

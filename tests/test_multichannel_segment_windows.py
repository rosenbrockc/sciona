import numpy as np
import pytest
from sciona.atoms.riemannian_bci.signal_processing.segment_windows import window_signal_segments


def test_window_order_boundaries_and_segment_indices():
    a=np.arange(22).reshape(2,11);b=np.arange(100,114).reshape(2,7)
    before=a.copy(),b.copy()
    windows,ids=window_signal_segments([a,b],4,4)
    np.testing.assert_array_equal(windows,np.stack([a[:,:4],a[:,4:8],b[:,:4]]))
    np.testing.assert_array_equal(ids,[0,0,1])
    windows[0,0,0]=-99
    np.testing.assert_array_equal(a,before[0]);np.testing.assert_array_equal(b,before[1])


def test_overlap_is_explicit_and_never_crosses_segment_boundary():
    a=np.arange(12,dtype=float).reshape(2,6)
    windows,ids=window_signal_segments([a,a+100],4,2)
    np.testing.assert_array_equal(windows,np.stack([a[:,:4],a[:,2:6],a[:,:4]+100,a[:,2:6]+100]))
    np.testing.assert_array_equal(ids,[0,0,1,1])


@pytest.mark.parametrize('segments,window,hop',[
    ([],4,4),([np.ones((2,3))],4,4),([np.ones((2,5)),np.ones((3,5))],4,4),
    ([np.ones((2,5))],True,4),([np.ones((2,5))],4,0),([np.full((2,5),np.nan)],4,4),
])
def test_invalid_segments_rejected(segments,window,hop):
    with pytest.raises(ValueError):window_signal_segments(segments,window,hop)

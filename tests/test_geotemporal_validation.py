import numpy as np
import pytest
from sciona.geotemporal_validation import forward_splits
from sciona.geotemporal_postprocess import smooth


def test_forward_gap_and_warmup():
    splits=forward_splits([0.,1.,3.,4.,6.,7.],[0,0,1,1,2,2],gap=2.)
    assert [(a.tolist(),b.tolist()) for a,b in splits]==[([0],[2,3]),([0,1,2],[4,5])]
    # Block zero is explicit fitting warmup, never mislabeled as OOF validation.
    assert sorted(np.concatenate([b for _,b in splits]))==[2,3,4,5]

@pytest.mark.parametrize('times,blocks,gap',[
    ([0.,2.,1.],[0,0,1],0.),([0.,1.],[0,1],2.),
    ([0.,0.],[0,1],0.),([0.,1.],[0,2],0.),
])
def test_invalid_forward_splits(times,blocks,gap):
    with pytest.raises(ValueError):forward_splits(times,blocks,gap=gap)


def test_hand_same_time_smoothing_and_clipping():
    points=[dict(x=0.,y=0.,time=1.),dict(x=1.,y=0.,time=1.),dict(x=0.,y=0.,time=2.)]
    np.testing.assert_allclose(smooth(points,[-2.,4.,100.],radius=2.,strength=.25),[1.,3.,100.])


def test_postprocess_permutation_and_no_neighbor():
    points=[dict(x=0.,y=0.,time=1.),dict(x=1.,y=0.,time=1.)]
    first=smooth(points,[2.,4.],radius=2.,strength=.5)
    assert first==smooth(points[::-1],[4.,2.],radius=2.,strength=.5)[::-1]
    assert smooth(points,[2.,4.],radius=.5,strength=1.)==[2.,4.]

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sciona import dfdc_convex_hull as implementation


def predictor_points():
    points=np.zeros((68,2),dtype=int)
    order=list(range(17))+list(range(26,16,-1))
    outline=[[4,4],[20,4],[20,18],[4,18]]+[[4,4]]*23
    points[order]=outline
    return points


def predictor_for(points):
    return lambda image,rectangle:SimpleNamespace(parts=lambda:[SimpleNamespace(x=int(x),y=int(y)) for x,y in points])


@pytest.mark.parametrize('draws,axis,keep_high', [
    ([.6,.6],0,True),([.5,.6],0,False),
    ([.6,.5],1,True),([.5,.5],1,False),
])
def test_independent_rectangle_centroid_and_half_selection(draws,axis,keep_high):
    image=np.full((25,29,3),83,dtype=np.uint8)
    expected=image.copy()
    rows,cols=np.indices(image.shape[:2])
    region=(rows>=4)&(rows<=18)&(cols>=4)&(cols<=20)
    coordinate,midpoint=(rows,11) if axis==0 else (cols,12)
    region &= coordinate>=midpoint if keep_high else coordinate<midpoint
    expected[region]=0
    values=iter(draws)
    with patch.object(implementation,'random',SimpleNamespace(random=lambda:next(values))):
        result=implementation.blackout_convex_hull(image,lambda image:[object()],predictor_for(predictor_points()))
    assert result is None
    np.testing.assert_array_equal(image,expected)
    assert list(values)==[]


@pytest.mark.parametrize('failure', ['empty_detector','predictor_error','out_of_bounds'])
def test_source_failure_is_noop_without_random_draws(failure):
    image=np.full((25,29,3),83,dtype=np.uint8)
    detector=lambda image:[object()]
    points=predictor_points()
    predictor=predictor_for(points)
    if failure=='empty_detector':detector=lambda image:[]
    if failure=='predictor_error':
        def predictor(image,rectangle):raise RuntimeError('synthetic failure')
    if failure=='out_of_bounds':predictor=predictor_for(points+100)
    with patch.object(implementation,'random',SimpleNamespace(random=lambda:pytest.fail('unexpected random draw'))):
        implementation.blackout_convex_hull(image,detector,predictor)
    assert np.all(image==83)

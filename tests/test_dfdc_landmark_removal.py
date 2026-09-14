from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from sciona import dfdc_landmark_removal as removal


def landmarks():
    return np.array([[5,5],[13,5],[9,10],[6,14],[12,14]],dtype=np.int32)


def test_eye_mask_matches_independent_manhattan_geometry():
    image=np.full((21,21,3),81,dtype=np.uint8)
    rows,cols=np.indices(image.shape[:2])
    distance=np.abs(rows-5)+np.maximum(np.maximum(5-cols,cols-13),0)
    expected=image.copy();expected[distance<=3]=0
    actual=removal.remove_eyes(image,landmarks())
    np.testing.assert_array_equal(actual,expected)
    assert np.all(image==81)


@pytest.mark.parametrize('operation', ['remove_eyes','remove_nose','remove_mouth'])
def test_zero_iterations_fills_entire_mask(operation):
    points=np.array([[5,5],[6,5],[5,7],[5,9],[6,9]],dtype=np.int32)
    image=np.full((13,13,3),81,dtype=np.uint8)
    assert not getattr(removal,operation)(image,points).any()
    assert image.all()


@pytest.mark.parametrize('draws,expected_operation', [
    ([.6],'remove_eyes'),([.5,.6],'remove_mouth'),
    ([.5,.5,.6],'remove_nose'),([.5,.5,.5],None),
])
def test_branch_order_strict_threshold_and_draw_count(draws,expected_operation):
    sequence=iter(draws)
    image=np.full((21,21,3),81,dtype=np.uint8)
    expected=image if expected_operation is None else getattr(removal,expected_operation)(image,landmarks())
    with patch.object(removal,'random',SimpleNamespace(random=lambda:next(sequence))):
        actual=removal.remove_landmark(image,landmarks())
    np.testing.assert_array_equal(actual,expected)
    assert list(sequence)==[]
    if expected_operation is None:assert actual is image

"""Synthetic confidence-tail selection contracts."""
import numpy as np
import pytest
from sciona.santander_pseudo_labels import select_pseudo_labels


def test_extreme_tails_and_original_position_alignment():
    scores=np.array([.5,.9,.1,.8,.3,.2])
    old=scores.copy()
    indices,labels=select_pseudo_labels(scores,positives=2,negatives=1)
    np.testing.assert_array_equal(indices,[1,2,3])
    np.testing.assert_array_equal(labels,[1,0,1])
    np.testing.assert_array_equal(scores,old)


def test_monotonic_scores_preserve_selected_rows_and_labels():
    scores=np.array([3.,7.,1.,5.])
    for a,b in zip(select_pseudo_labels(scores,positives=1,negatives=2),
                   select_pseudo_labels(scores*4+9,positives=1,negatives=2)):
        np.testing.assert_array_equal(a,b)


def test_ties_inside_selected_region_allowed_but_boundary_ties_rejected():
    indices,labels=select_pseudo_labels([0.,0.,2.,3.],positives=1,negatives=2)
    np.testing.assert_array_equal(indices,[0,1,3])
    np.testing.assert_array_equal(labels,[0,0,1])
    with pytest.raises(ValueError,match='Ambiguous'):
        select_pseudo_labels([0.,0.,2.,3.],positives=1,negatives=1)


@pytest.mark.parametrize('positive,negative,expected',[(0,2,[0,0]),(2,0,[1,1]),(1,1,[0,1])])
def test_entire_population_and_one_sided_selection(positive,negative,expected):
    indices,labels=select_pseudo_labels([0.,1.],positives=positive,negatives=negative)
    np.testing.assert_array_equal(indices,[0,1])
    np.testing.assert_array_equal(labels,expected)


@pytest.mark.parametrize('scores,p,n',[([1,2],2,1),([1,2],0,0),([1,2],True,0),
    ([1,2],-1,1),([1,float('nan')],1,0),([[1,2]],1,0),(['a'],1,0)])
def test_invalid_selection(scores,p,n):
    with pytest.raises(ValueError):select_pseudo_labels(scores,positives=p,negatives=n)

"""Synthetic final rank semantics; no full ensemble execution claim."""
import numpy as np
import pytest
from sciona.santander_blend import blend_neural_tree


def test_hand_rank_weights_and_ties():
    result=blend_neural_tree([.2,.8,.8],[30.,10.,20.])
    np.testing.assert_allclose(result,(2.1*np.array([1.,2.5,2.5])+[3.,1.,2.])/3.1)


def test_monotonic_rescaling_does_not_change_blend():
    np.testing.assert_array_equal(blend_neural_tree([1,3,2],[8,2,5]),
        blend_neural_tree([10,30,20],[64,4,25]))


def test_query_permutation_and_singleton():
    n=np.array([1.,4.,2.]);t=np.array([3.,2.,1.]);order=[2,0,1]
    np.testing.assert_array_equal(blend_neural_tree(n[order],t[order]),blend_neural_tree(n,t)[order])
    np.testing.assert_array_equal(blend_neural_tree([.9],[.01]),[1.])


@pytest.mark.parametrize('n,t',[([],[]),([1],[1,2]),([float('nan')],[1]),([[1]],[[2]]),(['a'],[1])])
def test_invalid_scores(n,t):
    with pytest.raises(ValueError):blend_neural_tree(n,t)

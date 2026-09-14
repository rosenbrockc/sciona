import math
import numpy as np
import pytest
from sciona.otto_tfidf import fit_weighting


def test_hand_smoothed_idf_and_l2_normalization():
    fit=fit_weighting([[2,0,0],[1,3,0]],smooth_idf=True,sublinear_tf=False,norm='l2')
    expected=np.array([1,math.log(3/2)+1,math.log(3)+1])
    np.testing.assert_allclose(fit.idf,expected)
    value=np.array([2,1,1])*expected
    np.testing.assert_allclose(fit.transform([[2,1,1]])[0],value/np.sqrt(sum(value**2)))


def test_unsmoothed_and_sublinear_l1():
    fit=fit_weighting([[2,0],[1,3]],smooth_idf=False,sublinear_tf=True,norm='l1')
    expected=np.array([1+math.log(4),(1+math.log(2))*(1+math.log(3))])
    np.testing.assert_allclose(fit.transform([[4,3]])[0],expected/expected.sum())


def test_zero_rows_and_large_values():
    fit=fit_weighting([[1,1],[1,1]],smooth_idf=True,sublinear_tf=False,norm='l2')
    result=fit.transform([[0,0],[1e300,1e300]])
    np.testing.assert_array_equal(result[0],0)
    np.testing.assert_allclose(result[1],[1/math.sqrt(2)]*2)


def test_fit_reference_detached_and_query_independent():
    reference=np.array([[1.,0],[0,2.]])
    fit=fit_weighting(reference,smooth_idf=True,sublinear_tf=False,norm='none')
    before=fit.idf.copy();a=fit.transform([[1,3]])
    reference[:]=999
    b=fit.transform([[1,3],[1000,1000]])
    np.testing.assert_array_equal(a,b[:1]);np.testing.assert_array_equal(fit.idf,before)
    assert not fit.idf.flags.writeable


@pytest.mark.parametrize('problem',['unsmoothed_absent','bad_flag','bad_norm','width','negative'])
def test_invalid_weighting(problem):
    with pytest.raises(ValueError):
        if problem=='unsmoothed_absent':fit_weighting([[1,0]],smooth_idf=False,sublinear_tf=False,norm='l2')
        elif problem=='bad_flag':fit_weighting([[1]],smooth_idf=1,sublinear_tf=False,norm='l2')
        elif problem=='bad_norm':fit_weighting([[1]],smooth_idf=True,sublinear_tf=False,norm='l3')
        else:
            fit=fit_weighting([[1]],smooth_idf=True,sublinear_tf=False,norm='l2')
            fit.transform([[1,2]] if problem=='width' else [[-1]])


def test_sublinear_fractional_counts_rejected():
    with pytest.raises(ValueError):fit_weighting([[.1]],smooth_idf=True,sublinear_tf=True,norm='l2')
    fit=fit_weighting([[1]],smooth_idf=True,sublinear_tf=True,norm='l2')
    with pytest.raises(ValueError):fit.transform([[.1]])

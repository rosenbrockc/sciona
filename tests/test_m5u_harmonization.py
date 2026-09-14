import numpy as np
import pytest
from sciona.m5u_harmonization import restore


def test_median_totals_move_toward_level_mean_all_quantiles_share_factor():
    # Native median totals are 10 and 30, so the target is 20.
    native=np.array([[[2.,3.,15.,8.]],[[4.,6.,30.,12.]],[[8.,12.,60.,20.]]])
    factors=np.array([.2,.2,.1,1.])
    before=native*factors
    actual=restore(before,[1,1,2,12],factors,[.1,.5,.9])
    expected=native*np.array([1.7,1.7,23/30,1.])
    np.testing.assert_allclose(actual,expected)
    np.testing.assert_array_equal(before,native*factors)


def test_each_horizon_is_adjusted_independently_and_full_adjustment_equalizes():
    values=np.array([[[10.,30.],[60.,20.]]])
    result=restore(values,[1,2],[1.,1.],[.5],adjustment=1.)
    np.testing.assert_allclose(result,[[[20.,20.],[40.,40.]]])


def test_levels_above_nine_only_restore_units_without_clipping():
    values=np.array([[[-2.,4.]],[[1.,2.]]])
    np.testing.assert_allclose(restore(values,[10,12],[.5,1.],[.9,.5]),values/[.5,1.])


def test_zero_median_and_inconsistent_level_multiplier_reject():
    with pytest.raises(ValueError,match='nonzero'):
        restore(np.zeros((1,1,1)),[1],[1.],[.5])
    with pytest.raises(ValueError,match='multiplier'):
        restore(np.ones((1,1,2)),[1,1],[1.,2.],[.5])

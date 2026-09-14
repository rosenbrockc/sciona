import math
import numpy as np
import pytest
from sciona.otto_preprocessing import representation,fit_scaling


def test_documented_fixed_transforms():
    x=[[0.,1.],[3.,8.]]
    np.testing.assert_array_equal(representation(x,kind='raw'),x)
    np.testing.assert_allclose(representation(x,kind='log1p'),[[0,math.log(2)],[math.log(4),math.log(9)]])
    np.testing.assert_allclose(representation(x,kind='sqrt_offset'),[[math.sqrt(3/8),math.sqrt(11/8)],[math.sqrt(27/8),math.sqrt(67/8)]])
    np.testing.assert_array_equal(representation(x,kind='raw_zero'),[[0,1,1,0],[3,8,0,0]])
    np.testing.assert_allclose(representation(x,kind='raw_zero_log'),[[0,1,1,0,0,math.log(2)],[3,8,0,0,math.log(4),math.log(9)]])


@pytest.mark.parametrize('ddof,scale',[(0,1.),(1,math.sqrt(2))])
def test_independent_scaling_and_constant_column(ddof,scale):
    fitted=fit_scaling([[1.,7.],[3.,7.]],kind='raw',ddof=ddof)
    np.testing.assert_allclose(fitted.mean,[2,7])
    np.testing.assert_allclose(fitted.scale,[scale,1])
    np.testing.assert_allclose(fitted.transform([[4,8]]),[[2/scale,1]])


def test_log_before_scaling():
    fitted=fit_scaling([[0.],[3.]],kind='log1p',ddof=0)
    np.testing.assert_allclose(fitted.transform([[1.]]),[[0.]],atol=1e-15)
    assert fitted.mean[0]==pytest.approx(math.log(2))


def test_reference_only_and_detached_statistics():
    reference=np.array([[1.,2.],[3.,4.]])
    fitted=fit_scaling(reference,kind='raw',ddof=0)
    a=fitted.transform([[5.,6.]])
    reference[:]=999
    b=fitted.transform([[5.,6.],[1000.,1000.]])
    np.testing.assert_array_equal(a,b[:1])
    np.testing.assert_array_equal(fitted.mean,[2,3])
    assert not fitted.mean.flags.writeable and not fitted.scale.flags.writeable


@pytest.mark.parametrize('values',[[],[[float('nan')]], [[-1]], [[1],[1,2]]])
def test_invalid_numeric_inputs(values):
    with pytest.raises(ValueError):representation(values,kind='raw')


def test_scaling_controls_and_width_rejection():
    with pytest.raises(ValueError):fit_scaling([[1]],kind='raw',ddof=1)
    with pytest.raises(ValueError):fit_scaling([[1],[2]],kind='raw',ddof=True)
    fitted=fit_scaling([[1],[2]],kind='raw',ddof=0)
    with pytest.raises(ValueError):fitted.transform([[1,2]])

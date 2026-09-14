import numpy as np
import pytest
from sciona.m5u_loss import pinball,weighted_scaled


def test_asymmetric_scalar_oracle_and_weight_normalization():
    a=np.array([1.,4.,3.]);p=np.array([2.,2.,3.]);w=np.array([1.,3.,0.]);s=np.array([2.,4.,1.])
    assert pinball(a,p,.2)==pytest.approx((.8+.4)/3)
    assert weighted_scaled(a,p,w,s,.2)==pytest.approx((.8/2+3*.4/4)/4)
    assert weighted_scaled(a,p,w*9,s,.2)==pytest.approx(weighted_scaled(a,p,w,s,.2))


def test_quantile_complement_symmetry():
    a=np.array([-2.,1.,9.]);p=np.array([5.,1.,3.])
    assert pinball(a,p,.1)==pytest.approx(pinball(-a,-p,.9))


@pytest.mark.parametrize('q',[0,1,-.1,np.nan,True])
def test_invalid_quantile_rejects(q):
    with pytest.raises(ValueError):pinball([1.],[2.],q)


def test_undefined_scaling_and_broadcasting_reject():
    for weights,scale in [([0.],[1.]),([1.],[0.]),([-1.],[1.])]:
        with pytest.raises(ValueError):weighted_scaled([1.],[2.],weights,scale)
    with pytest.raises(ValueError):pinball([1.,2.],[[1.,2.]])

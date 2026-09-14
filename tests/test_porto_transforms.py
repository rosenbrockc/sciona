import copy
import numpy as np
import pytest
from scipy.special import erfinv
from sciona.porto_transforms import rank_gauss_population,swap_noise


def test_rank_gauss_scalar_oracle_binary_passthrough_and_constant():
    x=np.array([[3.,0.,7.],[1.,1.,7.],[2.,0.,7.]])
    result=rank_gauss_population(x,binary_columns=[1])
    np.testing.assert_allclose(result[:,0],[erfinv(2/3),-erfinv(2/3),0],atol=1e-15)
    np.testing.assert_array_equal(result[:,1],x[:,1]);np.testing.assert_array_equal(result[:,2],0.)
    np.testing.assert_array_equal(x[:,0],[3,1,2])


def test_ties_are_equal_and_permutation_equivariant():
    x=np.array([[-2.],[-2.],[7.],[12.]])
    a=rank_gauss_population(x,binary_columns=[]);order=[3,0,2,1]
    b=rank_gauss_population(x[order],binary_columns=[])
    assert a[0,0]==a[1,0] and a[0,0]<a[2,0]<a[3,0]
    np.testing.assert_array_equal(b,a[order]);np.testing.assert_allclose(a.mean(axis=0),0,atol=1e-15)


def test_swap_noise_scalar_draw_oracle_and_column_support():
    x=np.full((5,3),-100.);reference=np.arange(18,dtype=float).reshape(6,3)
    rng=np.random.default_rng(7);mask=rng.random(x.shape)<.4;rows=rng.integers(6,size=x.shape)
    expected=x.copy()
    for row in range(5):
        for column in range(3):
            if mask[row,column]:expected[row,column]=reference[rows[row,column],column]
    actual=swap_noise(x,reference,probability=.4,rng=np.random.default_rng(7))
    np.testing.assert_array_equal(actual,expected)
    full=swap_noise(x,reference,probability=1.,rng=np.random.default_rng(7))
    for column in range(3):assert set(full[:,column])<=set(reference[:,column])
    np.testing.assert_array_equal(x,-100.)


def test_zero_noise_is_detached_and_preserves_generator_state():
    x=np.ones((2,3));rng=np.random.default_rng(4);state=copy.deepcopy(rng.bit_generator.state)
    result=swap_noise(x,x,probability=0,rng=rng);result[0,0]=9
    assert x[0,0]==1 and rng.bit_generator.state==state


@pytest.mark.parametrize('binary',[[0,0],[True],[3]])
def test_invalid_binary_declarations(binary):
    with pytest.raises(ValueError):rank_gauss_population(np.ones((3,2)),binary_columns=binary)


def test_nonbinary_and_nonfinite_inputs_rejected():
    with pytest.raises(ValueError):rank_gauss_population([[0.],[2.]],binary_columns=[0])
    with pytest.raises(ValueError):swap_noise([[float('nan')]],[[1.]],probability=.1,rng=np.random.default_rng(0))
    with pytest.raises(ValueError):swap_noise([[1.]],[[1.,2.]],probability=.1,rng=np.random.default_rng(0))
